import metrics


def init(data, state):
    data['tableProjects'] = {
        "columns": metrics.dataset_and_project_columns,
        "data": []
    }
    data['tableDatasets'] = {
        "columns": metrics.dataset_and_project_columns,
        "data": []
    }
    data['tableImages'] = {
        "columns": metrics.image_columns,
        "data": []
    }
    data['tableClasses'] = {
        "columns": metrics.table_classes_columns,
        "data": []
    }
    data['line_chart_options'] = {
        "title": "Line chart",
        "showLegend": True
    }
    data['lineChartSeries'] = []


def calculate_overall_metrics(api, task_id, gts, pred, dst_project_name, method, iou_threshold, score_threshold):
    images_pd_data = metrics.calculate_image_mAP(gts, pred, method, target_class=None, iou=iou_threshold,
                                                 score=score_threshold, need_rez=False)
    datasets_pd_data = metrics.calculate_dataset_mAP(gts, pred, method, target_class=None, iou=iou_threshold,
                                                     score=score_threshold)
    projects_pd_data, prj_rez = metrics.calculate_project_mAP(gts, pred, method, dst_project_name, target_class=None,
                                                              iou=iou_threshold, score=score_threshold)

    line_chart_series, table_classes = metrics.line_chart_builder(prj_rez['per_class'])
    line_chart_options = {"title": "Line chart", "showLegend": True}
    fields = [
        # {"field": "data.loading", "payload": False},
        {"field": "data.tableProjects", "payload": {"columns": metrics.dataset_and_project_columns,
                                                    "data": projects_pd_data}},
        {"field": "data.tableDatasets", "payload": {"columns": metrics.dataset_and_project_columns,
                                                    "data": datasets_pd_data}},
        {"field": "data.tableImages", "payload": {"columns": metrics.image_columns, "data": images_pd_data}},

        {"field": "data.tableClasses", "payload": {"columns": metrics.table_classes_columns, "data": table_classes}},
        {"field": "data.lineChartOptions", "payload": line_chart_options},
        {"field": "data.lineChartSeries", "payload": line_chart_series},
    ]
    api.app.set_fields(task_id, fields)
