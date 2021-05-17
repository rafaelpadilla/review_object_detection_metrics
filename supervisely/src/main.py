import os
import supervisely_lib as sly
import globals as g
import ui
import download_data as dd
from supervisely.src.confusion_matrix import confusion_matrix_py

import numpy as np


@g.my_app.callback("evaluate_button_click")
@sly.timeit
def evaluate_button_click(api: sly.Api, task_id, context, state, app_logger):
    # collect settings
    selected_classes = state['selectedClasses']
    percentage = state['samplePercent']
    iou_threshold = state['IoUThreshold']/100
    score_threshold = state['ScoreThreshold']/100
    dd.exec_download(selected_classes, percentage=percentage, confidence_threshold=score_threshold)

    print('dd.plt_boxes =', dd.plt_boxes.keys())
    gt = dd.plt_boxes['gt_images']
    det = dd.plt_boxes['pred_images']

    cm = confusion_matrix_py.confusion_matrix(gt_boxes=gt, det_boxes=det)
    print('confusion_matrix =', cm)
    # conf_matrx_columns, conf_matrx_data = confusion_matrix_py.convert_confusion_matrix_to_plt_format(cm)
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

    print(res_table)
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


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        # "modal.state.slyProjectId": project_id,  # @TODO: log more input envs
    })
    data = {}
    state = {}
    g.my_app.compile_template(g.root_source_dir)
    # init data for UI widgets
    ui.init(data, state)
    g.my_app.run(data=data, state=state)


# @TODO: check requirements - two files instead of one
# @TODO: disable class selection for conflicts
# @TODO: clear global unused requirements
if __name__ == "__main__":
    sly.main_wrapper("main", main)
