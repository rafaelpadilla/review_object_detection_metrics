# data="data.slyConfusionMatrix.data"
# options="{ selectable: true }"
# v-model="state.selected">

# from supervisely.src.confusion_matrix import confusion_matrix
from supervisely.src import download_data as dd
from supervisely.src.confusion_matrix import confusion_matrix_py


def init(data, state):
    # print('dd.plt_boxes =', dd.plt_boxes.keys())
    # gt = dd.plt_boxes['gt_images']
    # det = dd.plt_boxes['pred_images']
    #
    # cm = confusion_matrix_py.confusion_matrix(gt_boxes=gt, det_boxes=det)
    # print('confusion_matrix =', cm)
    # # conf_matrx_columns, conf_matrx_data = confusion_matrix_py.convert_confusion_matrix_to_plt_format(cm)
    # conf_matrx_columns_v2, conf_matrx_data_v2 = confusion_matrix_py.convert_confusion_matrix_to_plt_format_v2(cm)
    #
    # print('conf_matrx_data_v2 =', conf_matrx_data_v2)
    #
    # diagonal_max = 0
    # max_value = 0
    # data = list()
    #
    # for i1, row in enumerate(conf_matrx_data_v2):
    #     tmp = []
    #     for i2, col in enumerate(row):
    #         tmp.append(dict(value=col))
    #     data.append(tmp)

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
