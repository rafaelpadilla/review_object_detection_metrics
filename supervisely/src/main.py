import os
import numpy as np
import supervisely_lib as sly
from algorithm import process, get_data, get_data_v1, dict2tuple, calculate_mAP
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


@app.callback("app-gui-template")
@sly.timeit
def app_gui_template(api: sly.Api, task_id, context, state, app_logger):
    # Table Data
    iou         = 0.5
    round_level = 4
    # method = MethodAveragePrecision.EVERY_POINT_INTERPOLATION
    method = MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION
    image_columns   = ['SRC_ID', 'DST_ID', "name", "TP", "FP", "Precision", "Recall", "AP"]
    dataset_columns = ["name", "TP", "FP", "Precision", "Recall", "AP"]

    source_list = get_data_v1(src_project, dst_project)
    source_list_np = np.array(source_list, dtype=object)
    del source_list
    src_ids = np.where(source_list_np[:, 0] == 'src')
    dst_ids = np.where(source_list_np[:, 0] == 'dst')
    src_list_np = source_list_np[src_ids]
    dst_list_np = source_list_np[dst_ids]
    del source_list_np

    # print(src_list_np[0])
    # print(dst_list_np[0])

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
    # print('datasets_pd_data =', datasets_pd_data)

    projects_pd_data = list()
    src_set_list = list()
    dst_set_list = list()
    for l in src_list_np[:, -1]:
        src_set_list.extend(l)
    for l in dst_list_np[:, -1]:
        dst_set_list.extend(l)
    # print('Project Stage!')
    prj_rez = calculate_mAP(src_set_list, dst_set_list, 0.5, method)
    rez_d = dict2tuple(prj_rez)
    current_data = [dst_project.name]
    current_data.extend(rez_d)
    projects_pd_data.append(current_data)
    # -------------------------------------------
    line_chart_series = []
    tableClasses = []
    line_chart_options = {
        "title": "Line chart",
        "showLegend": True
    }
    showInterpolatedPrecision = True,
    prj_viz_data = prj_rez['per_class']
    for classId, result in prj_viz_data.items():
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        method = result['method']
        if showInterpolatedPrecision:
            nrec  = []
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
        # line_chart_series.append(dict(name=classId+' interpolated', data=[[i, j] for i, j in zip(nrec, nprec)]))
        FP   = result['total FP']
        TP   = result['total TP']
        npos = result['total positives']
        AP = round(result['AP'], round_level)
        Recall = round(TP / npos, round_level)
        Precision = round(np.divide(TP, (FP + TP)), round_level)
        tableClasses.append([classId, TP, FP, npos, Recall, Precision, AP])

    tableClasses_columns = ['classId', 'TP', 'FP', 'npos', 'Recall', 'Precision', 'AP']
    fields = [
        {"field": "data.tableClasses", "payload": {"columns": tableClasses_columns, "data": tableClasses}},
        {"field": "data.lineChartOptions", "payload": line_chart_options},
        {"field": "data.lineChartSeries", "payload": line_chart_series},
    ]

    api.app.set_fields(task_id, fields)
    # --------------------------------------------
    # save report to file *.lnk (link to report)
    report_name = f"app-gui-template.lnk"
    local_path = os.path.join(app.data_dir, report_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app.app_url, file=text_file)
    remote_path = api.file.get_free_name(TEAM_ID, f"/reports/app-gui-template/{report_name}")
    report_name = sly.fs.get_file_name_with_ext(remote_path)
    file_info = api.file.upload(TEAM_ID, local_path, remote_path)
    report_url = api.file.get_url(file_info.id)
    api.task.set_output_report(task_id, file_info.id, file_info.name)

    fields = [
        {"field": "data.started", "payload": False},
        {"field": "data.loading", "payload": False},
        {"field": "data.tableProjects", "payload": {"columns": dataset_columns, "data": projects_pd_data}},
        {"field": "data.tableDatasets", "payload": {"columns": dataset_columns, "data": datasets_pd_data}},
        {"field": "data.tableImages", "payload": {"columns": image_columns, "data": images_pd_data}},
        {"field": "data.savePath", "payload": remote_path},
        {"field": "data.reportName", "payload": report_name},
        {"field": "data.reportUrl", "payload": report_url},
    ]

    api.app.set_fields(task_id, fields)


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
        "IoUmarker": {}
    }
    app.run(data=data, state=state, initial_events=[{"command": "app-gui-template"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)