import globals as g
from supervisely_lib.app.widgets.confusion_matrix import ConfusionMatrix


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
    data['CMTableImages'] = {}
    data['CMGallery'] = {}
    data['CMImageTableTitle'] = "Cell is not selected."
    data['CMImageTableDescription'] = "Cell is not selected."


confusion_matrix = ConfusionMatrix(api=g.api, task_id=g.task_id, v_model='data.slyConfusionMatrix')
