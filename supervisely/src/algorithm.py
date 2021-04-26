import matplotlib.pyplot as plt
import numpy as np
import os

from dotenv import load_dotenv
from dotenv import dotenv_values
from collections import namedtuple
import json
import yaml
import supervisely_lib as sly

import random
import pickle

import sys
sys.path.append('../../')
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, CoordinatesType
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from src.utils.enumerators import MethodAveragePrecision

# statistic data structure
result = namedtuple('Result', ['TP', 'FP', 'NPOS', 'Precision', 'Recall', 'AP'])
image_bbs = namedtuple('image_bbs', ['image', 'bbs'])

api: sly.Api = sly.Api.from_env()
app: sly.AppService = sly.AppService()

global TEAM_ID
TEAM_ID = int(os.environ['context.teamId'])


def prepare_data(src_project, dst_project, batch_size):
    tmp_list = list()
    for src_dataset, dst_dataset in zip(api.dataset.get_list(src_project.id), api.dataset.get_list(dst_project.id)):
        src_images = api.image.get_list(src_dataset.id)
        dst_images = api.image.get_list(dst_dataset.id)
        print('SRC =', src_dataset.id, src_dataset.name)
        print('DST =', dst_dataset.id, dst_dataset.name)
        for src_batch, dst_batch in zip(sly.batched(src_images, batch_size), sly.batched(dst_images, batch_size)):
            src_image_ids = [image_info.id for image_info in src_batch]
            dst_image_ids = [image_info.id for image_info in dst_batch]
            src_image_urls = [image_info.full_storage_url for image_info in src_batch]
            dst_image_urls = [image_info.full_storage_url for image_info in dst_batch]

            progress = sly.Progress("Annotations downloaded: ", len(src_image_ids))
            src_annotations = api.annotation.download_batch(src_dataset.id, src_image_ids,
                                                            progress_cb=progress.iters_done_report)
            dst_annotations = api.annotation.download_batch(dst_dataset.id, dst_image_ids)

            for idx, (src_annotation, dst_annotation) in enumerate(zip(src_annotations, dst_annotations)):
                img_gts_bbs = plt2bb(src_annotation)
                img_det_bbs = plt2bb(dst_annotation, bb_type=BBType.DETECTED)

                tmp_list.append([src_project.id, src_project.name,
                                 src_dataset.id, src_dataset.name,
                                 src_annotation.image_id, src_annotation.image_name,
                                 src_image_urls[idx], img_gts_bbs])

                tmp_list.append([dst_project.id, dst_project.name,
                                 dst_dataset.id, dst_dataset.name,
                                 dst_annotation.image_id, dst_annotation.image_name,
                                 dst_image_urls[idx], img_det_bbs])
    return tmp_list


def get_data_v1(existing_array, src_project: sly.project, dst_project: sly.project, batch_size=10, percentage=1) -> tuple:
    projects = {
        'src': src_project,
        'dst': dst_project
    }

    image_names = set(existing_array[:, 5]) if len(existing_array) != 0 else []
    result_list = list()
    image_num = 0
    for key, project in projects.items():
        bb_type = BBType.GROUND_TRUTH if key == 'src' else BBType.DETECTED
        datasets = api.dataset.get_list(project.id)
        for dataset in datasets:
            image_list = api.image.get_list(dataset.id)
            if key == 'src':
                image_num += len(image_list)
            sample_size = int(np.ceil(len(image_list)/100 * percentage))
            image_list_random_sample = random.sample(image_list, sample_size)
            for ix, batch in enumerate(sly.batched(image_list_random_sample, batch_size)):
                image_ids   = [image_info.id for image_info in batch]
                annotations = api.annotation.download_batch(dataset.id, image_ids)
                for idx, annotation in enumerate(annotations):
                    if batch[idx].name not in image_names:
                        img_bbs = plt2bb(annotation, bb_type=bb_type)
                        result_list.append([project.id, project.name,
                                            dataset.id, dataset.name,
                                            batch[idx].id, batch[idx].name,
                                            batch[idx].full_storage_url, img_bbs])

    result_list = np.vstack([existing_array, np.array(result_list, dtype=object)]) if len(existing_array) != 0 else np.array(result_list, dtype=object)
    return result_list, image_list


def get_data_v3(src_project: sly.project, dst_project: sly.project, batch_size=10, percentage=100):
    report_name = 'src_dst_annotations.pkl'
    local_path = os.path.join(app.data_dir, report_name)
    remote_path = f"/reports/mAP_Calculator/{dst_project.name}/{report_name}"

    if api.file.exists(TEAM_ID, remote_path):
        api.file.download(TEAM_ID, remote_path, local_path)
        with open(local_path, 'rb') as f:
            f = f.seek(0)
            data = pickle.load(f)
    else:
        tmp_list = prepare_data(src_project, dst_project, batch_size)
        data = np.array(tmp_list, dtype=object)
        sly.fs.ensure_base_path(local_path)
        with open(local_path, 'wb') as f:
            pickle.dump(data, f)
        file_info = api.file.upload(TEAM_ID, local_path, remote_path)

    prj_ids = list(set(data[:, 0]))
    src_prj_items = data[np.where(data[:, 0] == prj_ids[0])]
    dst_prj_items = data[np.where(data[:, 0] == prj_ids[1])]

    src_list = list()
    dst_list = list()

    for name in set(src_prj_items[:, 3]):
        src_ds_items = src_prj_items[src_prj_items[:, 3] == name]
        dst_ds_items = dst_prj_items[dst_prj_items[:, 3] == name]
        length = len(src_ds_items)
        sample_size = int(np.ceil(length / 100 * percentage))
        indexes = random.sample(range(length), sample_size)
        src_list.extend(src_ds_items[indexes])
        dst_list.extend(dst_ds_items[indexes])

    src_list = np.array(src_list)
    dst_list = np.array(dst_list)
    data = np.vstack([src_list, dst_list])
    image_num = len(src_prj_items)

    return data, image_num


def get_data_v4(existing_array, src_project: sly.project, dst_project: sly.project, batch_size=10,
                percentage=1) -> tuple:
    projects = {
        'src': src_project,
        'dst': dst_project
    }
    result_list = list()
    image_num = 0
    sample_ids = dict()
    image_names = set(existing_array[:, 5]) if len(existing_array) != 0 else []

    for key, project in projects.items():
        bb_type = BBType.GROUND_TRUTH if key == 'src' else BBType.DETECTED
        datasets = api.dataset.get_list(project.id)
        # print('bb_type =', bb_type)
        for dataset in datasets:
            # print('dataset.name =', dataset.id, dataset.name)
            image_list = api.image.get_list(dataset.id)
            image_list_length = len(image_list)
            # print('dataset image_list len =', image_list_length)
            if key == 'src':
                image_num += image_list_length
                sample_size = int(np.ceil(image_list_length / 100 * percentage))
                random_seq = range(image_list_length)
                random_sample_ids = list()
                while (1):
                    random_id = random.choice(random_seq)
                    if image_list[random_id].name not in image_names:
                        if random_id not in random_sample_ids:
                            random_sample_ids.append(random_id)
                        if len(random_sample_ids) >= sample_size:
                            break
                sample_ids[dataset.name] = random_sample_ids
            else:
                random_sample_ids = sample_ids[dataset.name]

            image_list_random_sample = [image_list[i] for i in random_sample_ids]

            for ix, batch in enumerate(sly.batched(image_list_random_sample, batch_size)):
                image_ids = [image_info.id for image_info in batch]
                annotations = api.annotation.download_batch(dataset.id, image_ids)
                for idx, annotation in enumerate(annotations):
                    if batch[idx].name not in image_names:
                        img_bbs = plt2bb(annotation, bb_type=bb_type)
                        result_list.append([project.id, project.name,
                                            dataset.id, dataset.name,
                                            batch[idx].id, batch[idx].name,
                                            batch[idx].full_storage_url, img_bbs])

    existing_array = np.vstack([existing_array, np.array(result_list, dtype=object)]) \
        if len(existing_array) != 0 else np.array(result_list, dtype=object)
    return existing_array, image_num


def plt2bb(batch_element, type_coordinates=CoordinatesType.ABSOLUTE, bb_type=BBType.GROUND_TRUTH, format=BBFormat.XYX2Y2):
    # type_coordinates = CoordinatesType.X : ABSOLUTE, RELATIVE
    # bb_type          = BBType.X          : GROUND_TRUTH, DETECTED
    # format           = BBFormat.X        : XYX2Y2, XYWH, PASCAL_XML, YOLO
    ret = []
    annotations = batch_element.annotation['objects']
    for ann in annotations:
        class_title = ann['classTitle']
        points = ann['points']['exterior']
        x1, y1 = points[0]
        x2, y2 = points[1]
        if x1 >= x2 or y1 >= y2:
            print(x1 >= x2, y1 >= y2, x1 >= x2 or y1 >= y2)
            continue

        width = batch_element.annotation['size']['width']
        height = batch_element.annotation['size']['height']
        confidence = None if bb_type == BBType.GROUND_TRUTH else ann['tags'][0]['value']
        bb = BoundingBox(image_name=batch_element.image_name, class_id=class_title, coordinates=(x1, y1, x2, y2),
                         type_coordinates=type_coordinates, img_size=(width, height), confidence=confidence,
                         bb_type=bb_type, format=format)
        ret.append(bb)
    return ret


def calculate_mAP(img_gts_bbs, img_det_bbs, iou, score, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION) -> list:
    score_filtered_detections = [bbox for bbox in img_det_bbs if bbox.get_confidence() >= score]
    return get_pascalvoc_metrics(img_gts_bbs, score_filtered_detections, iou, generate_table=True, method=method)


def dict2tuple(dictionary, target_class, round_level=4):
    false_positive, true__positive, num__positives = 0, 0, 0
    if target_class and target_class !='ALL':
        dict__ = dictionary['per_class'][target_class]
        false_positive = dict__['total FP']
        true__positive = dict__['total TP']
        num__positives = dict__['total positives']
        average_precision = round(dict__['AP'], round_level)
        recall = round(true__positive / num__positives, round_level) if num__positives != 0 else 0
        precision = round(np.divide(true__positive, (false_positive + true__positive)), round_level) \
            if false_positive + true__positive != 0 else 0
    else:
        try:
            for dict_ in dictionary['per_class']:
                dict__ = dictionary['per_class'][dict_]
                false_positive += dict__['total FP']
                true__positive += dict__['total TP']
                num__positives += dict__['total positives']
        except:
            return result(0, 0, 0, 0, 0, 0)
        average_precision = round(dictionary['mAP'], round_level)
        recall = round(true__positive / num__positives, round_level) if num__positives != 0 else 0
        precision = round(np.divide(true__positive, (false_positive + true__positive)), round_level) \
            if false_positive + true__positive != 0 else 0

    return result(true__positive, false_positive, num__positives, precision, recall, average_precision)


def calculate_image_mAP(src_list_np, dst_list_np, method, target_class=None, iou=0.5, score=0.01, need_rez=False):
    images_pd_data = list()
    full_logs = list()
    for src_image_info, dst_image_info in zip(src_list_np, dst_list_np):
        assert src_image_info[5] == dst_image_info[5], 'different images'

        rez = calculate_mAP(src_image_info[-1], dst_image_info[-1], iou, score, method)
        # print('Rez =', rez)

        rez_d = dict2tuple(rez, target_class)
        src_image_image_id     = src_image_info[4]
        dst_image_image_id     = dst_image_info[4]
        src_image_image_name   = src_image_info[5]
        src_image_link         = src_image_info[6]
        per_image_data = [src_image_image_id, dst_image_image_id,
            '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(src_image_link, src_image_image_name)]
        per_image_data.extend(rez_d)
        images_pd_data.append(per_image_data)
        full_logs.append(rez)
    if need_rez:
        return images_pd_data, full_logs
    else:
        return images_pd_data


def calculate_dataset_mAP(src_list_np, dst_list_np, method, target_class=None, iou=0.5, score=0.01):
    datasets_pd_data = list()
    dataset_results = []
    for ds_name in set(dst_list_np[:, 3]):
        src_dataset_ = src_list_np[np.where(src_list_np[:, 3] == ds_name)]
        dst_dataset_ = dst_list_np[np.where(dst_list_np[:, 3] == ds_name)]
        src_set_list = list()
        dst_set_list = list()
        for l in src_dataset_[:, -1]:
            src_set_list.extend(l)
        for l in dst_dataset_[:, -1]:
            dst_set_list.extend(l)
        # print('Dataset Stage')
        rez = calculate_mAP(src_set_list, dst_set_list, iou, score, method)
        rez_d = dict2tuple(rez, target_class)
        current_data = [ds_name]
        current_data.extend(rez_d)

        dataset_results.append(rez['per_class'])
        datasets_pd_data.append(current_data)
    return datasets_pd_data


def calculate_project_mAP(src_list_np, dst_list_np, method, dst_project, target_class=None, iou=0.5, score=0.01):
    projects_pd_data = list()
    src_set_list = list()
    dst_set_list = list()
    for l in src_list_np[:, -1]:
        src_set_list.extend(l)
    for l in dst_list_np[:, -1]:
        dst_set_list.extend(l)
    prj_rez = calculate_mAP(src_set_list, dst_set_list, iou, score, method)
    rez_d = dict2tuple(prj_rez, target_class)
    current_data = [dst_project.name]
    current_data.extend(rez_d)
    projects_pd_data.append(current_data)
    return projects_pd_data, prj_rez


def line_chart_builder(prj_viz_data, round_level=4, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION):
    line_chart_series = []
    table_classes = []

    for classId, result in prj_viz_data.items():
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')
        precision = result['precision']
        recall = result['recall']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        method = result['method']
        nrec = []
        nprec = []
        if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
            nrec, nprec = mrec, mpre
        elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
            # Remove duplicates, getting only the highest precision of each recall value
            for idx in range(len(mrec)):
                r = mrec[idx]
                if r not in nrec:
                    idxEq = np.argwhere(mrec == r)
                    nrec.append(r)
                    nprec.append(max([mpre[int(id)] for id in idxEq]))

        line_chart_series.append(dict(name=classId, data=[[i, j] for i, j in zip(recall, precision)]))
        FP = result['total FP']
        TP = result['total TP']
        npos = result['total positives']
        AP = round(result['AP'], round_level)
        Recall = round(TP / npos, round_level) if npos != 0 else 0
        Precision = round(np.divide(TP, (FP + TP)), round_level) if FP + TP != 0 else 0
        table_classes.append([classId, float(TP), float(FP), float(npos), float(Recall), float(Precision), float(AP)])
    return line_chart_series, table_classes
