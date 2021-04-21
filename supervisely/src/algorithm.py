import matplotlib.pyplot as plt
import numpy as np
import os

from dotenv import load_dotenv
from dotenv import dotenv_values
from collections import namedtuple
import json
import yaml
import supervisely_lib as sly

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


def get_data_v1(src_project: sly.project, dst_project: sly.project, batch_size=10, num_batches=1) -> tuple:
    projects = {
        'src': src_project,
        'dst': dst_project
    }
    result_list = list()
    for key, project in projects.items():
        bb_type = BBType.GROUND_TRUTH if key == 'src' else BBType.DETECTED
        datasets = api.dataset.get_list(project.id)
        for dataset in datasets:
            image_list = api.image.get_list(dataset.id)
            for idx, batch in enumerate(sly.batched(image_list, batch_size)):
                if idx == num_batches:
                    break
                image_ids   = [image_info.id for image_info in batch]
                annotations = api.annotation.download_batch(dataset.id, image_ids)
                for idx, annotation in enumerate(annotations):
                    img_bbs = plt2bb(annotation, bb_type=bb_type)
                    result_list.append([key, dataset.id, dataset.name, batch[idx].id, batch[idx].name, batch[idx].full_storage_url, img_bbs])
    return result_list


def plt2bb(batch_element,
           type_coordinates = CoordinatesType.ABSOLUTE,
           bb_type          = BBType.GROUND_TRUTH,
           format           = BBFormat.XYX2Y2):
    # type_coordinates = CoordinatesType.X : ABSOLUTE, RELATIVE
    # bb_type          = BBType.X          : GROUND_TRUTH, DETECTED
    # format           = BBFormat.X        : XYX2Y2, XYWH, PASCAL_XML, YOLO
    ret = []
    annotations = batch_element.annotation['objects']
    for ann in annotations:
        class_title = ann['classTitle']
        try:
            points = ann['points']['exterior']
        except:
            print('annotation = ', ann)
            break
        x1, y1 = points[0]
        x2, y2 = points[1]
        width  = batch_element.annotation['size']['width']
        height = batch_element.annotation['size']['height']
        confidence = None if bb_type == BBType.GROUND_TRUTH else ann['tags'][0]['value']
        bb = BoundingBox(image_name      = batch_element.image_name,
                         class_id        = class_title,
                         coordinates     = (x1, y1, x2, y2),
                         type_coordinates= type_coordinates,
                         img_size        = (width, height),
                         confidence      = confidence,
                         bb_type         = bb_type,
                         format          = format)
        ret.append(bb)
    return ret


def calculate_mAP(img_gts_bbs, img_det_bbs, iou, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION) -> list:
    return get_pascalvoc_metrics(
        img_gts_bbs, img_det_bbs, iou, generate_table=True,
        method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION
    )


def dict2tuple(dictionary, round_level=4):
    FP = 0
    TP = 0
    npos = 0
    try:
        for dict_ in dictionary['per_class']:
            dict__ = dictionary['per_class'][dict_]
            FP += dict__['total FP']
            TP += dict__['total TP']
            npos += dict__['total positives']
    except:
        return result(0, 0, 0, 0, 0, 0)
    AP = round(dictionary['mAP'], round_level)
    Recall = round(TP / npos, round_level) if npos != 0 else 0
    Precision = round(np.divide(TP, (FP + TP)), round_level) if FP + TP != 0 else 0
    return result(TP, FP, npos, Precision, Recall, AP)


def calculate_image_mAP(src_list_np, dst_list_np, method):
    images_pd_data = list()
    for src_image_info, dst_image_info in zip(src_list_np, dst_list_np):
        rez = calculate_mAP(src_image_info[-1], dst_image_info[-1], 0.5, method)
        rez_d = dict2tuple(rez)
        # src_image_prj_type     = src_image_info[0]
        # dst_image_prj_type     = dst_image_info[0]
        # src_image_dataset_id   = src_image_info[1]
        # dst_image_dataset_id   = dst_image_info[1]
        # src_image_dataset_name = src_image_info[2]
        # dst_image_dataset_name = dst_image_info[2]
        src_image_image_id     = src_image_info[3]
        dst_image_image_id     = dst_image_info[3]
        src_image_image_name   = src_image_info[4]
        # dst_image_image_name   = dst_image_info[4]
        src_image_link         = src_image_info[5]
        # dst_image_link         = dst_image_info[5]

        per_image_data = [
            src_image_image_id, dst_image_image_id,
            '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(src_image_link, src_image_image_name)]
        per_image_data.extend(rez_d)
        images_pd_data.append(per_image_data)
    return images_pd_data


def calculate_dataset_mAP(src_list_np, dst_list_np, method):
    datasets_pd_data = list()
    dataset_results = []
    for ds_name in set(dst_list_np[:, 2]):
        src_dataset_ = src_list_np[np.where(src_list_np[:, 2] == ds_name)]
        dst_dataset_ = dst_list_np[np.where(dst_list_np[:, 2] == ds_name)]
        src_set_list = list()
        dst_set_list = list()
        for l in src_dataset_[:, -1]:
            src_set_list.extend(l)
        for l in dst_dataset_[:, -1]:
            dst_set_list.extend(l)
        # print('Dataset Stage')
        rez = calculate_mAP(src_set_list, dst_set_list, 0.5, method)
        rez_d = dict2tuple(rez)
        current_data = [ds_name]
        current_data.extend(rez_d)

        dataset_results.append(rez['per_class'])
        datasets_pd_data.append(current_data)
    return datasets_pd_data


def calculate_project_mAP(src_list_np, dst_list_np, method, dst_project):
    projects_pd_data = list()
    src_set_list = list()
    dst_set_list = list()
    for l in src_list_np[:, -1]:
        src_set_list.extend(l)
    for l in dst_list_np[:, -1]:
        dst_set_list.extend(l)
    prj_rez = calculate_mAP(src_set_list, dst_set_list, 0.5, method)
    rez_d = dict2tuple(prj_rez)
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


def process(src_project, dst_project, ious=[0.5, 0.75], round_level=4):
    """ The function receives project identifiers and receives data from the server
    according to these identifiers, calculates metrics and returns the result as lists of
    structured data like:

            tuple(name, IOU05('TP', 'FP', 'Precision', 'Recall', 'AP'), IOU075('TP', 'FP', 'Precision', 'Recall', 'AP'))

    Args:
        src_project (sly.project): sly.project object for source project.
        dst_project (sly.project): sly.project object for source project.
        ious (list)              : list of IOU thresholds for evaluation
        round_level (int)        : pretty parameter for floating point values

    Returns:
        tuple: The return value: project_mAP, dataset_mAP, image_mAP
    """
    # round_level: round(target_value, round_level): round(0.123456, round_level) --> 0.1235
    # ious: list of IOU thresholds
    # storage lists for images, datasets, projects
    image_mAP   = []  # image_name   + mAP_05 + mAP_075
    dataset_mAP = []  # dataset_name + mAP_05 + mAP_075
    project_mAP = []  # project_name + mAP_05 + mAP_075

    project_gts_bbs = []
    project_det_bbs = []

    for src, dst in zip(api.dataset.get_list(src_project.id), api.dataset.get_list(dst_project.id)):
        src_images = api.image.get_list(src.id)
        dst_images = api.image.get_list(dst.id)

        dataset_gts_bbs = []
        dataset_det_bbs = []

        for src_batch, dst_batch in zip(sly.batched(src_images, batch_size=10),
                                        sly.batched(dst_images, batch_size=10)):
            src_image_ids   = [image_info.id for image_info in src_batch]
            src_image_names = [image_info.name for image_info in src_batch]
            dst_image_ids   = [image_info.id for image_info in dst_batch]
            dst_image_names = [image_info.name for image_info in dst_batch]

            src_annotations = api.annotation.download_batch(src.id, src_image_ids)
            dst_annotations = api.annotation.download_batch(dst.id, dst_image_ids)
            assert len(src_annotations) == len(dst_annotations), \
                'Lenghst of src_annotations and dst_annotations must be the same!'
            for src_annotation, dst_annotation in zip(src_annotations, dst_annotations):
                img_gts_bbs = plt2bb(src_annotation)
                img_det_bbs = plt2bb(dst_annotation, bb_type=BBType.DETECTED)
                dataset_gts_bbs.extend(img_gts_bbs)
                dataset_det_bbs.extend(img_det_bbs)
                image_mAPs = []
                for iou in ious:
                    dict_res = get_pascalvoc_metrics(
                        img_gts_bbs, img_det_bbs, iou, generate_table=True,
                        method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION
                    )
                    image_mAPs.append(dict_res)

                image_mAP.append(
                    struct(
                        name=src_annotation.image_name,
                        mAP_05=dict2tuple(image_mAPs[0], round_level),
                        mAP_075=dict2tuple(image_mAPs[1], round_level)
                    )
                )

        project_gts_bbs.extend(dataset_gts_bbs)
        project_det_bbs.extend(dataset_det_bbs)

        mAPs = []
        for iou in ious:
            dict_res = get_pascalvoc_metrics(
                dataset_gts_bbs, dataset_det_bbs, iou, generate_table=True,
                method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION
            )
            mAPs.append(dict_res)

        dataset_mAP.append(
            struct(
                name=src.name,
                mAP_05=dict2tuple(mAPs[0], round_level),
                mAP_075=dict2tuple(mAPs[1], round_level)
            )
        )

    mAPs = []
    for iou in ious:
        dict_res = get_pascalvoc_metrics(
            project_gts_bbs, project_det_bbs, iou, generate_table=True,
            method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION
        )
        mAPs.append(dict_res)

    project_mAP.append(
        struct(
            name=src_project.name,
            mAP_05=dict2tuple(mAPs[0], round_level),
            mAP_075=dict2tuple(mAPs[1], round_level)
        )
    )
    return project_mAP, dataset_mAP, image_mAP
