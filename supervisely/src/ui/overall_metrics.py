# from supervisely.app.widgets.sly_table import SlyTable
from widgets.sly_table import SlyTable
import globals as g
import metrics

prj_sly_table = SlyTable(g.api, g.task_id, "data.tableProjects", g.dataset_and_project_columns)
dataset_sly_table = SlyTable(g.api, g.task_id, "data.tableDatasets", g.dataset_and_project_columns)


def init(data, state):
    data['tableProjects'] = {
        "columns": g.dataset_and_project_columns,
        "data": []
    }
    data['tableDatasets'] = {
        "columns": g.dataset_and_project_columns,
        "data": []
    }
    data['tableImages'] = {
        "columns": g.image_columns,
        "data": []
    }


def calculate_overall_metrics(api, task_id, gts, pred, dst_project_name, method, iou_threshold, score_threshold):
    datasets_pd_data = metrics.calculate_dataset_mAP(gts, pred, method, target_class=None, iou=iou_threshold,
                                                     score=score_threshold)
    projects_pd_data, prj_rez = metrics.calculate_project_mAP(gts, pred, method, dst_project_name, target_class=None,
                                                              iou=iou_threshold, score=score_threshold)

    prj_sly_table.set_data(projects_pd_data)
    prj_sly_table.update()
    dataset_sly_table.set_data(datasets_pd_data)
    dataset_sly_table.update()

