import numpy as np

from algorithm import *
from confusion_matrix import confusion_matrix, convert_to_numbers, convert_confusion_matrix_to_plt_format, \
    convert_confusion_matrix_to_plt_format_v2
from src.utils.enumerators import MethodAveragePrecision
import random

from _confusion_matrix_.confusion_matrix_class import ConfusionMatrix

api: sly.Api = sly.Api.from_env()
app: sly.AppService = sly.AppService()

TASK_ID = int(os.environ['TASK_ID'])
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
src_project_id = int(os.environ['modal.state.slySrcProjectId'])
dst_project_id = int(os.environ['modal.state.slyDstProjectId'])

src_project = app.public_api.project.get_info_by_id(src_project_id)
if src_project is None:
    raise RuntimeError(f"Project id={src_project_id} not found")

dst_project = app.public_api.project.get_info_by_id(dst_project_id)
if dst_project is None:
    raise RuntimeError(f"Project id={dst_project_id} not found")

cm_class = ConfusionMatrix(src_project=src_project, dst_project=dst_project)

meta = app.public_api.project.get_meta(dst_project_id)
round_level = 4
iou = 0.5
method = MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION

# #@TODO fix initial_percentage level
initial_percentage = 1
_confusion_matrix_ = {}


def convert_annotation(np_array: np.ndarray, bb_type=BBType.GROUND_TRUTH):
    bboxes_list = list()
    if len(np_array.shape) == 2:
        for image in np_array:
            # print(image[-1])
            tmp = image.copy()
            tmp[-1] = encoder(BoundingBox, image[-1], bb_type=bb_type)
            bboxes_list.append(tmp)
    else:
        tmp = np_array.copy()
        tmp[-1] = encoder(BoundingBox, np_array[-1], bb_type=bb_type)
        bboxes_list.append(tmp)
    return np.array(bboxes_list, dtype=np.object)


@app.callback("app-gui-template")
@sly.timeit
def app_gui_template(api: sly.Api, task_id, context, state, app_logger):
    # Table Data
    image_columns = ['SRC_ID', 'DST_ID', "name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]
    dataset_columns = ["name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]

    cm_class.download_project_annotations(percentage=initial_percentage)

    cm_class.active_image_set['src'] = convert_annotation(cm_class.downloaded_data['src'], bb_type=BBType.GROUND_TRUTH)
    cm_class.active_image_set['dst'] = convert_annotation(cm_class.downloaded_data['dst'], bb_type=BBType.DETECTED)

    # print('src_data = ', cm_class.active_image_set['src'][-2:])
    # print('dst_data = ', cm_class.active_image_set['dst'][-2:])

    images_pd_data = calculate_image_mAP(cm_class.active_image_set['src'], cm_class.active_image_set['dst'], method)
    # print('images_pd_data =', images_pd_data)
    datasets_pd_data = calculate_dataset_mAP(cm_class.active_image_set['src'], cm_class.active_image_set['dst'], method)
    projects_pd_data, prj_rez = calculate_project_mAP(cm_class.active_image_set['src'], cm_class.active_image_set['dst'], method, dst_project)

    # Statistic + lineCharts by classes
    prj_viz_data = prj_rez['per_class']
    line_chart_series, table_classes = line_chart_builder(prj_viz_data)
    line_chart_options = {
        "title": "Line chart",
        "showLegend": True
    }
    table_classes_columns = ['className', 'TP', 'FP', 'npos', 'Recall', 'Precision', 'AP']
    fields = [
        {"field": "data.tableClasses", "payload": {"columns": table_classes_columns, "data": table_classes}},
        {"field": "data.lineChartOptions", "payload": line_chart_options},
        {"field": "data.lineChartSeries", "payload": line_chart_series},
    ]
    api.app.set_fields(task_id, fields)
    # --------------------------------------------
    # Google-like table
    all_line = ['ALL']
    [all_line.append(value) for value in projects_pd_data[0][1:]]
    table_classes.append(all_line)
    table_data = list()
    for tuple_ in table_classes:
        table_data.append(dict(className=tuple_[0], TP=tuple_[1], FP=tuple_[2], npos=tuple_[3],
                               Recall=tuple_[4], Precision=tuple_[5], AP=tuple_[6], tag='null'))
    fields = [
        {"field": "data.tableClassesExtended", "payload": table_data},
        {"field": "data.totalImagesCount", "payload": cm_class.image_num}
    ]
    api.app.set_fields(task_id, fields)
    fields = [
        # {"field": "data.started", "payload": False},
        {"field": "data.loading", "payload": False},
        {"field": "data.tableProjects", "payload": {"columns": dataset_columns, "data": projects_pd_data}},
        {"field": "data.tableDatasets", "payload": {"columns": dataset_columns, "data": datasets_pd_data}},
        {"field": "data.tableImages", "payload": {"columns": image_columns, "data": images_pd_data}},
    ]
    api.app.set_fields(task_id, fields)


@app.callback("evaluate")
@sly.timeit
def evaluate(api: sly.Api, task_id, context, state, app_logger):
    global cm_class

    state_percentage = state['samplePercent']
    iou_threshold = state['IoUThreshold'] / 100
    score_threshold = state['ScoreThreshold'] / 100
    print("!!! samplePercent =", state_percentage)

    if state_percentage != cm_class.current_percentage:
        cm_class.download_project_annotations(state_percentage)

        src = cm_class.downloaded_data['src']
        dst = cm_class.downloaded_data['dst']

        src_list = list()
        dst_list = list()

        for name in set(src[:, 3]):
            src_ds_items = src[src[:, 3] == name]
            dst_ds_items = dst[dst[:, 3] == name]
            length = len(src_ds_items)
            sample_size = int(np.ceil(length / 100 * state_percentage))
            indexes = random.sample(range(length), sample_size)
            src_list.extend(src_ds_items[indexes])
            dst_list.extend(dst_ds_items[indexes])

        cm_class.active_image_set['src'] = convert_annotation(np.array(src_list, dtype=np.object), bb_type=BBType.GROUND_TRUTH)
        cm_class.active_image_set['dst'] = convert_annotation(np.array(dst_list, dtype=np.object), bb_type=BBType.DETECTED)

    # print('cm_class.active_image_set[src] =', cm_class.active_image_set['src'])

    if cm_class.active_image_set['src'] == [] or cm_class.active_image_set['src'] == np.array([]):
        cm_class.download_project_annotations(percentage=initial_percentage)
        cm_class.active_image_set['src'] = convert_annotation(cm_class.downloaded_data['src'], bb_type=BBType.GROUND_TRUTH)
        cm_class.active_image_set['dst'] = convert_annotation(cm_class.downloaded_data['dst'], bb_type=BBType.DETECTED)

    projects_pd_data, prj_rez = calculate_project_mAP(cm_class.active_image_set['src'], cm_class.active_image_set['dst'],
                                                      method, dst_project, iou=iou_threshold, score=score_threshold)
    prj_viz_data = prj_rez['per_class']
    _, table_classes = line_chart_builder(prj_viz_data)

    # Google-like table
    all_line = ['ALL']
    [all_line.append(value) for value in projects_pd_data[0][1:]]
    table_classes.append(all_line)
    table_data = list()
    for tuple_ in table_classes:
        table_data.append(
            dict(className=tuple_[0], TP=tuple_[1], FP=tuple_[2], npos=tuple_[3],
                 Recall=tuple_[4], Precision=tuple_[5], AP=tuple_[6], tag='null'))

    # calculate confusion matrix
    global _confusion_matrix_
    _confusion_matrix_ = confusion_matrix(cm_class.active_image_set['src'], cm_class.active_image_set['dst'],
                                          iou_threshold, score_threshold)
    # _confusion_matrix_ = cm_class.confusion_matrix(iou_threshold, score_threshold)
    conf_matrx_columns, conf_matrx_data = convert_confusion_matrix_to_plt_format(_confusion_matrix_)
    conf_matrx_columns_v2, conf_matrx_data_v2 = convert_confusion_matrix_to_plt_format_v2(_confusion_matrix_)

    print('conf_matrx_data_v2 =', conf_matrx_data_v2)

    diagonal_max = 0
    max_value = 0
    data = list() # np.zeros_like(conf_matrx_data_v2)

    for i1, row in enumerate(conf_matrx_data_v2):
        tmp = []
        for i2, col in enumerate(row):
            tmp.append(dict(value=col))
        data.append(tmp)
    #
    # for i in range(len(conf_matrx_data_v2)):
    #     if conf_matrx_data_v2[i][i]>diagonal_max:
    #         diagonal_max = conf_matrx_data_v2[i][i]
    #     for j in range(len(conf_matrx_data_v2)):
    #         if i != j:
    #             if conf_matrx_data_v2[i][j]>max_value:
    #                 max_value = conf_matrx_data_v2[i][j]
    # total_pred =  conf_matrx_data_v2[]
    # np.sum()
    cm_data = {
        "classes": conf_matrx_columns_v2,
        "diagonalMax": diagonal_max,
        "maxValue": max_value,
        "data": data
    }

    fields = [
        {"field": "data.tableClassesExtended", "payload": table_data},
        {"field": "data.totalImagesCount", "payload": cm_class.image_num},
        {"field": "data.slyTableConfusionMatrix", "payload": {"columns": conf_matrx_columns, "data": conf_matrx_data.tolist()}},
        {"field": "data.slyConfusionMatrix", "payload": {"data": cm_data}}
    ]
    api.app.set_fields(task_id, fields)


def expand_line(data, class_name):
    class_list = []
    for el in data:
        for box in el[-1]:
            if box.get_class_id() == class_name:
                class_list.append(el)
                break
    return np.array(class_list)


def expand_line_v2(data, class_name):
    class_list = []
    for box in data:
        if box.get_class_id() == class_name:
            class_list.append(box.get_class_id())
            break
    return np.array(class_list)


@app.callback("view_class")
@sly.timeit
def view_class(api: sly.Api, task_id, context, state, app_logger):

    class_name = state["selectedClassName"]
    iou_threshold = state["IoUThreshold"]
    score_threshold = state['ScoreThreshold']

    print('class_name =', class_name)
    print('state =', state)

    # fraction_src_list = []
    # [fraction_src_list.extend(image) for image in cm_class.active_image_set['src']]
    # fraction_dst_list = []
    # [fraction_dst_list.extend(image) for image in cm_class.active_image_set['dst']]

    if class_name != 'ALL':
        # print('fraction_src_list_np =', cm_class.active_image_set['src'])
        images_names_with_target_class = expand_line(cm_class.active_image_set['src'], class_name)
        # images_names_with_target_class = list()
        # for el in cm_class.active_image_set['src']:
        #     for box in el[-1]:
        #         if box.get_class_id() == class_name:
        #             images_names_with_target_class.append(el)
        #             break

        # single_class_dst_list = list()
        # for el in cm_class.active_image_set['dst']:
        #     for box in el[-1]:
        #         if box.get_class_id() == class_name:
        #             single_class_dst_list.append(el)
        #             break

        row_indexes = [line in images_names_with_target_class for line in cm_class.active_image_set['src'][:, 5]]
        single_class_src_list_np = cm_class.active_image_set['src'][row_indexes]
        single_class_dst_list_np = cm_class.active_image_set['dst'][row_indexes]

        # images_names_with_target_class = expand_line_v2(fraction_src_list, class_name)
        # print('images_names_with_target_class =', images_names_with_target_class)
        # row_indexes = [line in images_names_with_target_class for line in fraction_src_list_np[:, 5]]
        # row_indexes = np.array([id_ for id_, line in enumerate(fraction_src_list)
        #                         if line.get_image_name() in images_names_with_target_class], dtype=np.bool)
        # single_class_src_list_np = np.array(fraction_src_list)[row_indexes]
        # single_class_dst_list_np = np.array(fraction_dst_list)[row_indexes]

        single_class_src_list_np = single_class_src_list_np
        single_class_dst_list_np = single_class_dst_list_np
    else:
        single_class_src_list_np = np.array(cm_class.active_image_set['src'])
        single_class_dst_list_np = np.array(cm_class.active_image_set['dst'])

    # print('single_class_src_list_np =', single_class_src_list_np)
    # depending on IOU and Score thresholds
    class_images_pd_data, class_full_logs = calculate_image_mAP(single_class_src_list_np, single_class_dst_list_np,
                                                                method, target_class=class_name, iou=iou_threshold/100,
                                                                score=score_threshold/100, need_rez=True)
    fp_images = [item for item in class_images_pd_data if item[4] != 0]
    print('fp_images =', fp_images)
    projects_pd_data, prj_rez = calculate_project_mAP(single_class_src_list_np, single_class_dst_list_np, method,
                                                      dst_project, iou=iou_threshold/100, score=score_threshold/100)
    prj_viz_data = prj_rez['per_class']
    line_chart_series, table_classes = line_chart_builder(prj_viz_data)
    if class_name != 'ALL':
        for i in line_chart_series:
            if i['name'] == class_name:
                target_chart = [i]
                break
        for i in table_classes:
            if i[0] == class_name:
                target_table = [i]
    else:
        target_chart = line_chart_series
        target_table = table_classes

    line_chart_options = {
        "title": "Line chart",
        "showLegend": True
    }
    image_columns = ['SRC_ID', 'DST_ID', "name", "TP", "FP", 'NPOS', "Precision", "Recall", "AP"]
    table_classes_columns = ['className', 'TP', 'FP', 'npos', 'Recall', 'Precision', 'AP']
    fields = [
        {"field": "data.tableSingleClassImages", "payload": {"columns": image_columns, "data": class_images_pd_data}},
        {"field": "data.classExtraStats", "payload": {"columns": table_classes_columns, "data": target_table}},
        {"field": "data.lineChartOptions", "payload": line_chart_options},
        {"field": "data.classExtraChartSeries", "payload": target_chart},
    ]
    api.app.set_fields(task_id, fields)


@app.callback("show_images")
@sly.timeit
def show_images(api: sly.Api, task_id, context, state, app_logger):
    print('Show_images: ', state)
    selected_row_data = state["selection"]["selectedRowData"]
    if selected_row_data is not None and state["selection"]["selectedColumnName"] is not None:
        keys = [key for key in selected_row_data]
        if 'SRC_ID' not in keys:
            return
        image_id_1 = selected_row_data['SRC_ID']
        image_id_2 = selected_row_data['DST_ID']
    else:
        return

    ann_1 = api.annotation.download(image_id_1)
    ann_2 = api.annotation.download(image_id_2)

    content = {
        "projectMeta": {
            "classes": meta['classes'],
            "tags": []
        },
        "annotations": {
            "ann_1": {
                "url": api.image.get_info_by_id(image_id_1).full_storage_url,
                "figures": ann_1.annotation['objects'],
                "info": {"title": "original"}
            },
            "ann_2": {
                "url": api.image.get_info_by_id(image_id_2).full_storage_url,
                "figures": ann_2.annotation['objects'],
                "info": {"title": "detection"}
            },
        },
        "layout": [["ann_1"], ["ann_2"]]
    }
    # <pre>{{data.previewContent}}</pre> # to show code
    options = {
        "showOpacityInHeader": False,
        "opacity": 0.8,
        "fillRectangle": False,
        "syncViews": True,
        "syncViewsBindings": [["ann_1", "ann_2"]]
    }
    fields = [
        {"field": "data.previewContent", "payload": content},
        {"field": "data.previewOptions", "payload": options},
    ]
    api.app.set_fields(task_id, fields)


@app.callback("show_conf_mtrx_details")
@sly.timeit
def show_conf_mtrx_details(api: sly.Api, task_id, context, state, app_logger):
    row_name = state['selection']['selectedRowData']['class_names']
    col_name = state['selection']['selectedColumnName']
    iou_threshold = state["IoUThreshold"]
    score_threshold = state['ScoreThreshold']
    it = _confusion_matrix_[col_name][row_name]
    image_names = set(it)
    tmp_src, tmp_dst = [], []
    image_columns = ['SRC_ID', 'DST_ID', "name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]
    for l in image_names:
        tmp_src.extend(src_list_np[src_list_np[:, 5] == l])
        tmp_dst.extend(dst_list_np[dst_list_np[:, 5] == l])
    images_pd_data = calculate_image_mAP(np.array(tmp_src), np.array(tmp_dst), method,
                                         iou=iou_threshold/100, score=score_threshold/100)
    # dst_list_np
    fields = [
        {"field": "data.confusionMatrixImageTable", "payload": {"columns": image_columns, "data": images_pd_data}}
    ]
    api.app.set_fields(task_id, fields)


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": TEAM_ID,
        "WORKSPACE_ID": WORKSPACE_ID,
        "SRC_PROJECT_ID": src_project,
        "DST_PROJECT_ID": dst_project
    })

    data = {
        "projectId": "",
        "projectName": "",
        "projectPreviewUrl": "",
        "progress": 0,
        "progressCurrent": 0,
        "progressTotal": 0,
        "clickedCell": "not clicked",
        "table": {"columns": [], "data": []},
        "selection": {},
        "cellToImages": {"columns": [], "data": []},
        "previewContent": {"projectMeta": {}, "projectMeta": {}},
        "tableProjects": {"columns": [], "data": []},
        "tableDatasets": {"columns": [], "data": []},
        "tableImages": {"columns": [], "data": []},
        "loading": True,
        "classExtraStats": {"columns": [], "data": []},
        "tableClassesExtended": [],
        "tableSingleClassImages": {"columns": [], "data": []},
        "classExtraChartSeries": [],
        "confusionMatrix": {},
        "slyConfusionMatrix": {},
        "confusionMatrixImageTable": {}
    }
    state = {
        "selection": {},
        "selectedClassName": "",
    }
    app.run(data=data, state=state, initial_events=[{"command": "app-gui-template"}]) #


if __name__ == "__main__":
    sly.main_wrapper("main", main)
