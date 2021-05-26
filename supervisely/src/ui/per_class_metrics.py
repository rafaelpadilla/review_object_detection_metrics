import numpy as np
import supervisely_lib as sly
import globals as g
import metrics


def init(data, state):
    data['perClassExtendedTable'] = []
    data['perClassTable'] = {"columns": metrics.table_classes_columns, "data": []}
    data['perClassLineChartOptions'] = {"title": "Precision/Recall curve", "showLegend": True}
    data['perClassLineChartSeries'] = []
    data['perClassSingleImagesTable'] = {"columns": metrics.image_columns, "data": []}
    data['perClass'] = {}
    data['perClassGalleryTitle'] = 'Please, select row from ImageTable.'
    note_text = '''There may be differences between the Class table and the Image table. 
    Such a situation is possible due to since when considering one single image, we do not pay 
    attention to other class errors in other images from the set, but when evaluating the entire set, 
    these errors are immediately accessible.'''
    data['notification'] = {
        "content": "##### {}".format(note_text),
        "options": {
            "name": "Important Note",
            "type": "note"
            }
    }
    data['perClassSelectedClsTitle'] = 'Class is not selected'


def calculate_per_classes_metrics(api, task_id, src_list, dst_list, dst_project_name, method,
                                  iou_threshold, score_threshold):
    projects_pd_data, prj_rez = metrics.calculate_project_mAP(src_list=src_list, dst_list=dst_list, method=method,
                                                              dst_project_name=dst_project_name,
                                                              iou=iou_threshold, score=score_threshold)
    # Statistic + lineCharts by classes
    prj_viz_data = prj_rez['per_class']
    line_chart_series, table_classes = metrics.line_chart_builder(prj_viz_data)
    # --------------------------------------------
    # Google-like table
    all_line = ['ALL']
    [all_line.append(value) for value in projects_pd_data[0][1:]]
    table_classes.append(all_line)
    table_data = list()
    for tuple_ in table_classes:
        table_data.append(dict(className=tuple_[0], TP=tuple_[1], FP=tuple_[2], npos=tuple_[3],
                               Recall=tuple_[4], Precision=tuple_[5], AP=tuple_[6], tag='null'))
    line_chart_options = {"title": "Precision/Recall curve", "showLegend": True}
    fields = [
        {"field": "data.perClassExtendedTable", "payload": table_data},
        # {"field": "data.perClassTable", "payload": {"columns": metrics.table_classes_columns,
        #                                             "data": table_classes}},
        {"field": "data.perClassLineChartOptions", "payload": line_chart_options},
        {"field": "data.perClassLineChartSeries", "payload": line_chart_series}
    ]
    api.app.set_fields(task_id, fields)


def expand_line(data, class_name):
    image_names = []
    for image in data:
        for bbox in image[-1]:
            if bbox.get_class_id() == class_name:
                image_names.append(image[1])
                break
    return image_names


def selected_class_metrics(api, task_id, src_list, dst_list, class_name, dst_project, iou_threshold, score_threshold):
    if class_name != 'ALL':
        images_names_with_target_class = list(set(expand_line(src_list, class_name) + expand_line(dst_list, class_name)))
        print('images_names_with_target_class =', images_names_with_target_class)
        row_indexes1 = [id_ for id_, line in enumerate(src_list) if line[1] in images_names_with_target_class]
        row_indexes2 = [id_ for id_, line in enumerate(dst_list) if line[1] in images_names_with_target_class]
        row_indexes = list(set(row_indexes1 + row_indexes2))
        print('row_indexes =', row_indexes)
        single_class_src_list_np = [src_list[id_] for id_ in row_indexes]
        single_class_dst_list_np = [dst_list[id_] for id_ in row_indexes]

    else:
        single_class_src_list_np = src_list
        single_class_dst_list_np = dst_list

    class_images_pd_data, \
    class_full_logs = metrics.calculate_image_mAP(single_class_src_list_np, single_class_dst_list_np,
                                                  metrics.MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                                  target_class=class_name,
                                                  iou=iou_threshold, score=score_threshold,
                                                  need_rez=True)
    projects_pd_data, \
    prj_rez = metrics.calculate_project_mAP(single_class_src_list_np, single_class_dst_list_np,
                                            metrics.MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                            dst_project, iou=iou_threshold, score=score_threshold)
    prj_viz_data = prj_rez['per_class']
    line_chart_series, table_classes = metrics.line_chart_builder(prj_viz_data)

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

    perClassSelectedClsTitle = 'Selected class is "{}"'.format(class_name)
    fields = [
        {"field": "data.perClassSingleImagesTable", "payload": {"columns": metrics.image_columns,
                                                                "data": class_images_pd_data}},
        # {"field": "data.perClassTable", "payload": {"columns": metrics.table_classes_columns,
        #                                             "data": target_table}},  # projects_pd_data
        {"field": "data.perClassLineChartSeries", "payload": target_chart},
        {"field": "data.perClassSelectedClsTitle", "payload": perClassSelectedClsTitle}
    ]
    api.app.set_fields(task_id, fields)
