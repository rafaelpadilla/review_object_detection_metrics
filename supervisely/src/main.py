import os
import numpy as np
import supervisely_lib as sly
from algorithm import process, get_data_v1, dict2tuple, calculate_image_mAP, calculate_dataset_mAP, calculate_project_mAP, line_chart_builder
from src.utils.enumerators import MethodAveragePrecision
from random import randrange
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


@app.callback("app-gui-template")
@sly.timeit
def app_gui_template(api: sly.Api, task_id, context, state, app_logger):
    # Table Data
    image_columns   = ['SRC_ID', 'DST_ID', "name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]
    dataset_columns = ["name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]

    source_list = get_data_v1(src_project, dst_project)
    source_list_np = np.array(source_list, dtype=object)
    del source_list
    src_ids = np.where(source_list_np[:, 0] == 'src')
    dst_ids = np.where(source_list_np[:, 0] == 'dst')
    src_list_np = source_list_np[src_ids]
    dst_list_np = source_list_np[dst_ids]
    del source_list_np

    images_pd_data = calculate_image_mAP(src_list_np, dst_list_np, method)
    datasets_pd_data = calculate_dataset_mAP(src_list_np, dst_list_np, method)
    projects_pd_data, prj_rez = calculate_project_mAP(src_list_np, dst_list_np, method, dst_project)
    # -------------------------------------------
    # Statistic + lineCharts by classes
    prj_viz_data = prj_rez['per_class']
    line_chart_series, table_classes = line_chart_builder(prj_viz_data)
    line_chart_options = {
        "title": "Line chart",
        "showLegend": True
    }
    table_classes_columns = ['className', 'TP', 'FP', 'npos', 'Recall', 'Precision', 'AP']
    print('table_classes =', table_classes)
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
            dict(className=tuple_[0],
                TP=tuple_[1],
                FP=tuple_[2],
                npos=tuple_[3],
                Recall=tuple_[4],
                Precision=tuple_[5],
                AP=tuple_[6],
                tag='null'))
    fields = [
        {"field": "data.tableClassesExtended", "payload": tableData},
        {"field": "data.sliderValue", "payload": 1}
    ]
    api.app.set_fields(task_id, fields)

    # --------------------------------------------
    # save report to file *.lnk (link to report)
    # report_name = f"app-gui-template.lnk"
    # local_path = os.path.join(app.data_dir, report_name)
    # sly.fs.ensure_base_path(local_path)
    # with open(local_path, "w") as text_file:
    #     print(app.app_url, file=text_file)
    # remote_path = api.file.get_free_name(TEAM_ID, f"/reports/app-gui-template/{report_name}")
    # report_name = sly.fs.get_file_name_with_ext(remote_path)
    # file_info = api.file.upload(TEAM_ID, local_path, remote_path)
    # report_url = api.file.get_url(file_info.id)
    # api.task.set_output_report(task_id, file_info.id, file_info.name)

    fields = [
        {"field": "data.started", "payload": False},
        {"field": "data.loading", "payload": False},
        {"field": "data.tableProjects", "payload": {"columns": dataset_columns, "data": projects_pd_data}},
        {"field": "data.tableDatasets", "payload": {"columns": dataset_columns, "data": datasets_pd_data}},
        {"field": "data.tableImages", "payload": {"columns": image_columns, "data": images_pd_data}},
        # {"field": "data.savePath", "payload": remote_path},
        # {"field": "data.reportName", "payload": report_name},
        # {"field": "data.reportUrl", "payload": report_url},
    ]
    api.app.set_fields(task_id, fields)


@app.callback("test")
@sly.timeit
def test(api: sly.Api, task_id, context, state, app_logger):
    percent = state["selection"]["sliderValue"]
    # recalculate table
    if percent == 0:
        return
    fields = [
        {"field": "data.sliderValue", "payload": percent}
    ]
    # api.app.set_fields(task_id, fields)


@app.callback("recalculate")
@sly.timeit
def recalculate(api: sly.Api, task_id, context, state, app_logger):
    print('sliderValue =', state["selection"]["sliderValue"])
    sliderValue = state["selection"]["sliderValue"] # data['sliderValue']
    fraction_source_list = get_data_v1(src_project, dst_project, num_batches=sliderValue)
    fraction_source_list_np = np.array(fraction_source_list, dtype=object)
    del fraction_source_list
    src_ids = np.where(fraction_source_list_np[:, 0] == 'src')
    dst_ids = np.where(fraction_source_list_np[:, 0] == 'dst')
    fraction_src_list_np = fraction_source_list_np[src_ids]
    fraction_dst_list_np = fraction_source_list_np[dst_ids]
    del fraction_source_list_np

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
            dict(className=tuple_[0],
                TP=tuple_[1],
                FP=tuple_[2],
                npos=tuple_[3],
                Recall=tuple_[4],
                Precision=tuple_[5],
                AP=tuple_[6],
                tag='null'))
    fields = [
        {"field": "data.tableClassesExtended", "payload": tableData},
        {"field": "data.sliderValue", "payload": sliderValue}
    ]
    api.app.set_fields(task_id, fields)
    for element in tableData:
        print(element['AP'])


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
                "figures": ann_1.annotation['objects']
            },
            "ann_2": {
                "url": api.image.get_info_by_id(image_id_2).full_storage_url,
                "figures": ann_2.annotation['objects']
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
        "loading": True
    }
    state = {
        "selection": {},
    }
    app.run(data=data, state=state, initial_events=[{"command": "app-gui-template"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)