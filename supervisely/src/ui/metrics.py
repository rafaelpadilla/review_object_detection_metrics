import numpy as np
from collections import namedtuple
import sys
import supervisely_lib as sly
import utils

sys.path.append('../../')
from src.bounding_box import BoundingBox, BBType, BBFormat
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from src.utils.enumerators import MethodAveragePrecision

result = namedtuple('Result', ['TP', 'FP', 'NPOS', 'Precision', 'Recall', 'AP'])

table_classes_columns = ['className', 'TP', 'FP', 'npos', 'Recall', 'Precision', 'AP']
image_columns = ['SRC_ID', 'DST_ID', "name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]
dataset_and_project_columns = ["name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]


def init(data, state):
    data['tableProjects'] = {
        "columns": dataset_and_project_columns,
        "data": []
    }
    data['tableDatasets'] = {
        "columns": dataset_and_project_columns,
        "data": []
    }
    data['tableImages'] = {
        "columns": image_columns,
        "data": []
    }
    data['tableClasses'] = {
        "columns": table_classes_columns,
        "data": []
    }
    data['line_chart_options'] = {
        "title": "Line chart",
        "showLegend": True
    }
    data['lineChartSeries'] = []


def dict2tuple(dictionary, target_class, round_level=4):
    false_positive, true__positive, num__positives = 0, 0, 0
    if target_class and target_class != 'ALL':
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


def calculate_mAP(img_gts_bbs, img_det_bbs, iou, score,
                  method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION) -> list:
    score_filtered_detections = []
    for bbox in img_det_bbs:
        try:
            if bbox.get_confidence() >= score:
                score_filtered_detections.append(bbox)
        except:
            print(bbox)
            if bbox.get_confidence() >= score:
                score_filtered_detections.append(bbox)
    return get_pascalvoc_metrics(img_gts_bbs, score_filtered_detections, iou, generate_table=True, method=method)


def calculate_image_mAP(src_list, dst_list, method, target_class=None, iou=0.5, score=0.01, need_rez=False):
    images_pd_data = list()
    full_logs = list()
    matched = 0
    print('Lengths of sets =', len(src_list), len(dst_list))
    for src_image_info in src_list:
        for dst_image_info in dst_list:
            if src_image_info[1] == dst_image_info[1]:
                matched += 1
                rez = calculate_mAP(src_image_info[-1], dst_image_info[-1], iou, score, method)
                rez_d = dict2tuple(rez, target_class)
                src_image_image_id = src_image_info[0]
                dst_image_image_id = dst_image_info[0]
                src_image_image_name = src_image_info[1]
                src_image_link = src_image_info[2]
                per_image_data = [src_image_image_id, dst_image_image_id,
                                  '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(
                                      src_image_link,
                                      src_image_image_name)]
                per_image_data.extend(rez_d)
                images_pd_data.append(per_image_data)
                full_logs.append(rez)
    print('processed  {} of (src={}, dst={})'.format(matched, len(src_list), len(dst_list)))
    if need_rez:
        return images_pd_data, full_logs
    else:
        return images_pd_data


def calculate_dataset_mAP(src_dict, dst_dict, method, target_class=None, iou=0.5, score=0.01):
    datasets_pd_data = list()
    dataset_results = []
    key_list = list(set([el[-2] for el in src_dict]))
    for dataset_key in key_list:
        src_set_list = []
        [src_set_list.extend(el[-1]) for el in src_dict if el[-2] == dataset_key]
        dst_set_list = []
        [dst_set_list.extend(el[-1]) for el in dst_dict if el[-2] == dataset_key]

        rez = calculate_mAP(src_set_list, dst_set_list, iou, score, method)
        rez_d = dict2tuple(rez, target_class)
        current_data = [dataset_key]
        current_data.extend(rez_d)

        dataset_results.append(rez['per_class'])
        datasets_pd_data.append(current_data)
    return datasets_pd_data


def calculate_project_mAP(src_list, dst_list, method, dst_project_name, target_class=None, iou=0.5, score=0.01):
    projects_pd_data = list()

    src_set_list = []
    [src_set_list.extend(el[-1]) for el in src_list]
    dst_set_list = []
    [dst_set_list.extend(el[-1]) for el in dst_list]

    prj_rez = calculate_mAP(src_set_list, dst_set_list, iou, score, method)
    rez_d = dict2tuple(prj_rez, target_class)
    current_data = [dst_project_name]
    current_data.extend(rez_d)
    projects_pd_data.append(current_data)
    return projects_pd_data, prj_rez


def line_chart_builder(prj_viz_data, round_level=4):
    line_chart_series = []
    table_classes = []

    for classId, res in prj_viz_data.items():
        if res is None:
            raise IOError(f'Error: Class {classId} could not be found.')
        precision = res['precision']
        recall = res['recall']
        fp = res['total FP']
        tp = res['total TP']
        npos = res['total positives']
        ap = round(res['AP'], round_level)
        # precision recall interpolation is removed: info - see src_backup/algorithm.py
        line_chart_series.append(dict(name=classId, data=[[i, j] for i, j in zip(recall, precision)]))
        recall = round(tp / npos, round_level) if npos != 0 else 0
        precision = round(np.divide(tp, (fp + tp)), round_level) if fp + tp != 0 else 0
        table_classes.append([classId, float(tp), float(fp), float(npos), float(recall), float(precision), float(ap)])
    return line_chart_series, table_classes


def calculate_metrics(api: sly.Api, task_id, src_list, dst_list, method, dst_project_name, target_class=None,
                      iou_threshold=0.5, score_threshold=0.01):
    encoder = BoundingBox

    gts = []
    pred = []
    dataset_names = {}
    for dataset_key in src_list:
        for gt_image, pr_image in zip(src_list[dataset_key], dst_list[dataset_key]):
            gt_boxes = utils.plt2bb(gt_image, encoder, bb_type=BBType.GROUND_TRUTH)
            pred_boxes = utils.plt2bb(pr_image, encoder, bb_type=BBType.DETECTED)

            if gt_image['dataset_id'] not in dataset_names:
                dataset_names[gt_image['dataset_id']] = api.dataset.get_info_by_id(gt_image['dataset_id']).name
            if pr_image['dataset_id'] not in dataset_names:
                dataset_names[pr_image['dataset_id']] = api.dataset.get_info_by_id(pr_image['dataset_id']).name

            gts.append([gt_image['image_id'], gt_image['image_name'], gt_image['full_storage_url'],
                        dataset_names[gt_image['dataset_id']], gt_boxes])
            pred.append([pr_image['image_id'], pr_image['image_name'], pr_image['full_storage_url'],
                         dataset_names[pr_image['dataset_id']], pred_boxes])

    images_pd_data = calculate_image_mAP(gts, pred, method, target_class=None, iou=iou_threshold,
                                         score=score_threshold, need_rez=False)
    datasets_pd_data = calculate_dataset_mAP(gts, pred, method, target_class=None, iou=iou_threshold,
                                             score=score_threshold)
    projects_pd_data, prj_rez = calculate_project_mAP(gts, pred, method, dst_project_name, target_class=None,
                                                      iou=iou_threshold, score=score_threshold)
    fields = [
        {"field": "data.loading", "payload": False},
        {"field": "data.tableProjects", "payload": {"columns": dataset_and_project_columns, "data": projects_pd_data}},
        {"field": "data.tableDatasets", "payload": {"columns": dataset_and_project_columns, "data": datasets_pd_data}},
        {"field": "data.tableImages", "payload": {"columns": image_columns, "data": images_pd_data}},
    ]
    api.app.set_fields(task_id, fields)

    prj_viz_data = prj_rez['per_class']
    line_chart_series, table_classes = line_chart_builder(prj_viz_data)
    line_chart_options = {
        "title": "Line chart",
        "showLegend": True
    }
    fields = [
        {"field": "data.tableClasses", "payload": {"columns": table_classes_columns, "data": table_classes}},
        {"field": "data.lineChartOptions", "payload": line_chart_options},
        {"field": "data.lineChartSeries", "payload": line_chart_series},
    ]
    api.app.set_fields(task_id, fields)

