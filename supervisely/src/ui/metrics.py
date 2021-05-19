import supervisely_lib as sly
import globals as g
import numpy as np
from collections import namedtuple
import sys
sys.path.append('../../')
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, CoordinatesType
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from src.utils.enumerators import MethodAveragePrecision

result = namedtuple('Result', ['TP', 'FP', 'NPOS', 'Precision', 'Recall', 'AP'])

table_classes_columns = ['className', 'TP', 'FP', 'npos', 'Recall', 'Precision', 'AP']
image_columns = ['SRC_ID', 'DST_ID', "name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]
dataset_and_project_columns = ["name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]


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


def calculate_image_mAP(src_list_np, dst_list_np, method, target_class=None, iou=0.5, score=0.01, need_rez=False):
    images_pd_data = list()
    full_logs = list()
    for src_image_info, dst_image_info in zip(src_list_np, dst_list_np):
        assert src_image_info[1] == dst_image_info[1], 'different images'

        rez = calculate_mAP(src_image_info[-1], dst_image_info[-1], iou, score, method)
        rez_d = dict2tuple(rez, target_class)
        src_image_image_id     = src_image_info[0]
        dst_image_image_id     = dst_image_info[0]
        src_image_image_name   = src_image_info[1]
        src_image_link         = src_image_info[2]
        per_image_data = [src_image_image_id, dst_image_image_id,
            '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>'.format(src_image_link, src_image_image_name)]
        per_image_data.extend(rez_d)
        images_pd_data.append(per_image_data)
        full_logs.append(rez)
    if need_rez:
        return images_pd_data, full_logs
    else:
        return images_pd_data


def calculate_project_mAP(src_list_np, dst_list_np, method, dst_project, target_class=None, iou=0.5, score=0.01):
    projects_pd_data = list()
    # print('src_list_np =', src_list_np)
    # src_set_list = src_list_np[:, -1].tolist()
    # dst_set_list = dst_list_np[:, -1].tolist()
    src_set_list = []

    dst_set_list = []
    [dst_set_list.extend(el) for el in dst_list_np[:, -1]]

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
        FP = result['total FP']
        TP = result['total TP']
        npos = result['total positives']
        AP = round(result['AP'], round_level)
        # precision recall interpolation is removed: info - see src_backup/algorithm.py
        line_chart_series.append(dict(name=classId, data=[[i, j] for i, j in zip(recall, precision)]))
        Recall = round(TP / npos, round_level) if npos != 0 else 0
        Precision = round(np.divide(TP, (FP + TP)), round_level) if FP + TP != 0 else 0
        table_classes.append([classId, float(TP), float(FP), float(npos), float(Recall), float(Precision), float(AP)])
    return line_chart_series, table_classes


def init(data, state):

    data['tableProjects'] = {
        "columns": dataset_and_project_columns,
        "data": []
    }
    data['tableDatasets'] = {
        "columns": dataset_and_project_columns,
        "data": [
            [
                212,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547865\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ],
            [
                213,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547867\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo-569380.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ],
            [
                214,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547869\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo-569380.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ],
            [
                215,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547866\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo-569380.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ],
            [
                216,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547868\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo-569380.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ]
        ]
    }
    data['tableImages'] = {
        "columns": image_columns,
        "data": [
            [
                212,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547865\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ],
            [
                213,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547867\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo-569380.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ],
            [
                214,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547869\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo-569380.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ],
            [
                215,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547866\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo-569380.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ],
            [
                216,
                "<a href=\"https://app.supervise.ly/app/images/5318/6947/68193/297005?page=1#image-140547868\" rel=\"noopener noreferrer\" target=\"_blank\">pexels-photo-569380.jpg</a>",
                "crowd",
                853,
                1280,
                3,
                100,
                0
            ]
        ]
    }

