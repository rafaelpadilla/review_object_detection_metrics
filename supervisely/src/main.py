import os
import numpy as np
import supervisely_lib as sly
from algorithm import get_data_v1, get_data_v3, get_data_v4, dict2tuple, calculate_image_mAP, \
    calculate_dataset_mAP, calculate_project_mAP, line_chart_builder
from src.utils.enumerators import MethodAveragePrecision
from random import randrange
import random
import pickle

from collections import defaultdict

app: sly.AppService = sly.AppService()

TASK_ID        = int(os.environ['TASK_ID'])
TEAM_ID        = int(os.environ['context.teamId'])
WORKSPACE_ID   = int(os.environ['context.workspaceId'])
src_project_id = int(os.environ['modal.state.slySrcProjectId'])
dst_project_id = int(os.environ['modal.state.slyDstProjectId'])

src_project = app.public_api.project.get_info_by_id(src_project_id)
if src_project is None:
    raise RuntimeError(f"Project id={src_project_id} not found")

dst_project = app.public_api.project.get_info_by_id(dst_project_id)
if dst_project is None:
    raise RuntimeError(f"Project id={dst_project_id} not found")

meta = app.public_api.project.get_meta(dst_project_id)
round_level = 4
iou = 0.5
method = MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION
# method = MethodAveragePrecision.EVERY_POINT_INTERPOLATION

# source_list_np, image_num = get_data_v3(src_project, dst_project, batch_size=50)
global existing_array, fraction_src_list_np, fraction_dst_list_np

existing_array, image_num = get_data_v4(np.array([], dtype=object), src_project, dst_project,
                                        batch_size=10, percentage=5)

# with open('../notebooks/existing_files.pkl', 'rb') as file:
#     existing_array = pickle.load(file)
# image_num = int(len(existing_array)/2)

src_ids = np.where(existing_array[:, 0] == src_project.id)
dst_ids = np.where(existing_array[:, 0] == dst_project.id)
src_list_np = existing_array[src_ids]
dst_list_np = existing_array[dst_ids]
fraction_src_list_np, fraction_dst_list_np = src_list_np, dst_list_np
# del src_ids, dst_ids


@app.callback("app-gui-template")
@sly.timeit
def app_gui_template(api: sly.Api, task_id, context, state, app_logger):
    # Table Data
    image_columns = ['SRC_ID', 'DST_ID', "name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]
    dataset_columns = ["name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]

    images_pd_data = calculate_image_mAP(src_list_np, dst_list_np, method)
    datasets_pd_data = calculate_dataset_mAP(src_list_np, dst_list_np, method)
    projects_pd_data, prj_rez = calculate_project_mAP(src_list_np, dst_list_np, method, dst_project)

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
    ALL_line = ['ALL']
    [ALL_line.append(value) for value in projects_pd_data[0][1:]]
    table_classes.append(ALL_line)
    tableData = list()
    for tuple_ in table_classes:
        tableData.append(
            dict(className=tuple_[0], TP=tuple_[1], FP=tuple_[2], npos=tuple_[3],
                 Recall=tuple_[4], Precision=tuple_[5], AP=tuple_[6], tag='null'))
    fields = [
        {"field": "data.tableClassesExtended", "payload": tableData},
        {"field": "data.totalImagesCount", "payload": image_num}
    ]
    api.app.set_fields(task_id, fields)
    fields = [
        {"field": "data.started", "payload": False},
        {"field": "data.loading", "payload": False},
        {"field": "data.tableProjects", "payload": {"columns": dataset_columns, "data": projects_pd_data}},
        {"field": "data.tableDatasets", "payload": {"columns": dataset_columns, "data": datasets_pd_data}},
        {"field": "data.tableImages", "payload": {"columns": image_columns, "data": images_pd_data}},
    ]
    api.app.set_fields(task_id, fields)


@app.callback("refresh")
@sly.timeit
def refresh(api: sly.Api, task_id, context, state, app_logger):
    global existing_array
    print(state)
    print('src_list_np =', src_list_np.shape)
    print('dst_list_np =', dst_list_np.shape)
    print('existing_array =', existing_array.shape)
    percentage = state['samplePercent']
    print('data loading...')
    updated_existing_array, image_num = get_data_v4(existing_array, src_project, dst_project, batch_size=20, percentage=percentage)
    existing_array = updated_existing_array
    print('updated_existing_array =', existing_array.shape)
    pass


@app.callback("evaluate")
@sly.timeit
def evaluate(api: sly.Api, task_id, context, state, app_logger):
    global existing_array, src_list_np, dst_list_np, image_num, fraction_src_list_np, fraction_dst_list_np

    print(locals())
    print("!!! ", state['samplePercent'])
    percentage = state['samplePercent']  # data['sliderValue']

    existing_array, image_num = get_data_v4(existing_array, src_project, dst_project,
                                            batch_size=20, percentage=percentage)

    src_list_np = existing_array[np.where(existing_array[:, 0] == src_project.id)]
    dst_list_np = existing_array[np.where(existing_array[:, 0] == dst_project.id)]

    src_list = list()
    dst_list = list()

    for name in set(src_list_np[:, 3]):
        src_ds_items = src_list_np[src_list_np[:, 3] == name]
        dst_ds_items = dst_list_np[dst_list_np[:, 3] == name]
        length = len(src_ds_items)
        sample_size = int(np.ceil(length / 100 * percentage))
        indexes = random.sample(range(length), sample_size)
        src_list.extend(src_ds_items[indexes])
        dst_list.extend(dst_ds_items[indexes])

    fraction_src_list_np = np.array(src_list)
    fraction_dst_list_np = np.array(dst_list)

    projects_pd_data, prj_rez = calculate_project_mAP(fraction_src_list_np, fraction_dst_list_np, method, dst_project)
    prj_viz_data = prj_rez['per_class']
    _, table_classes = line_chart_builder(prj_viz_data)
    # Google-like table
    ALL_line = ['ALL']
    [ALL_line.append(value) for value in projects_pd_data[0][1:]]
    table_classes.append(ALL_line)
    tableData = list()
    for tuple_ in table_classes:
        tableData.append(
            dict(className=tuple_[0], TP       =tuple_[1], FP=tuple_[2], npos=tuple_[3],
                 Recall   =tuple_[4], Precision=tuple_[5], AP=tuple_[6], tag ='null'))
    fields = [
        {"field": "data.tableClassesExtended", "payload": tableData},
        {"field": "data.totalImagesCount", "payload": image_num}
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


@app.callback("view_class")
@sly.timeit
def view_class(api: sly.Api, task_id, context, state, app_logger):
    global existing_array, image_num, src_list_np, dst_list_np
    class_name = state["selectedClassName"]
    iou_threshold = state["IoUThreshold"]
    score_threshold = state['ScoreThreshold']

    print('class_name =', class_name)
    print('state =', state)

    if class_name != 'ALL':
        images_names_with_target_class = expand_line(fraction_src_list_np, class_name)
        row_indexes = [line in images_names_with_target_class for line in fraction_src_list_np[:, 5]]
        single_class_src_list_np = fraction_src_list_np[row_indexes]
        single_class_dst_list_np = fraction_dst_list_np[row_indexes]
    else:
        single_class_src_list_np = fraction_src_list_np
        single_class_dst_list_np = fraction_dst_list_np

    # depending on IOU and Score thresholds
    class_images_pd_data, class_full_logs = calculate_image_mAP(single_class_src_list_np, single_class_dst_list_np,
                                                                method, target_class=class_name, iou=iou_threshold/100,
                                                                score=score_threshold/100, need_rez=True)

    fp_images = [item for item in class_images_pd_data if item[4] != 0]
    print('fp_images =', fp_images)

    projects_pd_data, prj_rez = calculate_project_mAP(fraction_src_list_np, single_class_dst_list_np, method,
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
    try:
        random_images_to_show = random.choice(fp_images)
    except:
        pass


@app.callback("show_images")
@sly.timeit
def show_images(api: sly.Api, task_id, context, state, app_logger):
    selected_row_data = state["selection"]["selectedRowData"]
    if selected_row_data is not None and state["selection"]["selectedColumnName"] is not None:
        keys = [key for key in selected_row_data]
        if 'SRC_ID' not in keys:
            return
        image_id_1 = selected_row_data['SRC_ID']
        image_id_2 = selected_row_data['DST_ID']
    else:
        return

    ann_1 = ann_info = api.annotation.download(image_id_1)
    ann_2 = ann_info = api.annotation.download(image_id_2)

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
    }
    state = {
        "selection": {},
        "selectedClassName": "",
    }
    app.run(data=data, state=state, initial_events=[{"command": "app-gui-template"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)