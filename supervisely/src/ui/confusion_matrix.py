import supervisely_lib as sly
import numpy as np
from supervisely.src.confusion_matrix import confusion_matrix_py


def init(data, state):
    state['selection'] = {}
    state['selected'] = {'rowClass': None, 'colClass': None}
    conf_matrx_columns_v2 = []
    diagonal_max = 0
    max_value = 0
    cm_data = []

    slyConfusionMatrix = {
        "classes": conf_matrx_columns_v2,
        "diagonalMax": diagonal_max,
        "maxValue": max_value,
        "data": cm_data
    }
    data['slyConfusionMatrix'] = slyConfusionMatrix
    data['confusionTableImages'] = {}
    data['confusionMatrixPreviewContent'] = {}
    data['confusionMatrixPreviewOptions'] = {}


def calculate_confusion_matrix(gt, det, iou_threshold, score_threshold, api: sly.Api, task_id):

    cm = confusion_matrix_py.confusion_matrix(gt_boxes=gt, det_boxes=det,
                                              iou_threshold=iou_threshold, score_threshold=score_threshold)
    conf_matrx_columns_v2, conf_matrx_data_v2 = confusion_matrix_py.convert_confusion_matrix_to_plt_format_v2(cm)
    diagonal_max = 0
    max_value = 0
    cm_data = list()
    np_table = np.array(conf_matrx_data_v2)
    a0 = np.sum(np_table, axis=0)  # столбец
    a1 = np.sum(np_table, axis=1)  # строка
    a0 = np.append(a0, 0)
    res_table = np.hstack((np_table, a1.reshape(-1, 1)))
    res_table = np.vstack((res_table, a0))

    conf_matrx_data_v2_extended = res_table.tolist()

    for i1, row in enumerate(conf_matrx_data_v2_extended):
        tmp = []
        for i2, col in enumerate(row):
            tmp.append(dict(value=int(col)))
            if i1 == i2 and col > diagonal_max:
                diagonal_max = col
            if i1 != i2 and col > max_value:
                max_value = col
        cm_data.append(tmp)

    slyConfusionMatrix = {
        "data": {
            "classes": conf_matrx_columns_v2,
            "diagonalMax": diagonal_max,
            "maxValue": max_value,
            "data": cm_data
        }
    }
    fields = [
        {"field": "data.slyConfusionMatrix", "payload": slyConfusionMatrix},
    ]
    api.app.set_fields(task_id, fields)
    return cm
