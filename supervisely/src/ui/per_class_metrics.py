import supervisely_lib as sly
import globals as g
import metrics
import settings
from widgets.sly_table import SlyTable
from widgets.compare_gallery import CompareGallery
import ui_utils

# from supervisely_lib.app.widgets.sly_table import SlyTable

# gallery_per_class = CompareGallery(g.task_id, g.api, 'data.perClass', g.aggregated_meta)
image_sly_table = None  # SlyTable(g.api, g.task_id, "data.perClassTable", g.image_columns)
gallery_per_class = None  # CompareGallery(g.task_id, g.api, 'data.perImage', g.aggregated_meta)


def init(data, state):
    state['perClassActiveStep'] = 1
    data['perClassExtendedTable'] = []
    data['perClassTable'] = {"columns": g.table_classes_columns, "data": []}
    data['perClassLineChartOptions'] = {"title": "Precision/Recall curve", "showLegend": True}
    data['perClassLineChartSeries'] = []
    data['perClassSingleImagesTable'] = {"columns": g.image_columns, "data": []}
    data['perClass'] = {}
    data['perClassGalleryTitle'] = 'Please, select row from ImageTable.'
    note_text = '''There may be differences between the Per Class Statistic table and the Image Statistic Table. 
    Such a situation is possible due to since when considering one single image, we do not pay 
    attention to selected class errors in other images from the set, but when evaluating the entire set, 
    these errors are immediately accessible.'''
    data['notification'] = {
        "content": "##### {}".format(note_text),
        "options": {
            "name": "Important Note",
            "type": "note"
        }
    }
    data['perClassSelectedClsTitle'] = 'Class is not selected'
    state['perClassActiveNames'] = []
    state['perClassShow1'] = False
    state['perClassShow2'] = False
    state['perClassShow3'] = False
    state['activeFigure'] = None


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
        single_class_src_list_np = {}
        single_class_dst_list_np = {}
        for dataset_name, dataset_items in src_list.items():
            images_names_with_target_class = list(
                set(expand_line(src_list[dataset_name], class_name) + expand_line(dst_list[dataset_name], class_name)))
            row_indexes1 = [id_ for id_, line in enumerate(src_list[dataset_name]) if
                            line[1] in images_names_with_target_class]
            row_indexes2 = [id_ for id_, line in enumerate(dst_list[dataset_name]) if
                            line[1] in images_names_with_target_class]
            row_indexes = list(set(row_indexes1 + row_indexes2))
            single_class_src_list_np[dataset_name] = [src_list[dataset_name][id_] for id_ in row_indexes]
            single_class_dst_list_np[dataset_name] = [dst_list[dataset_name][id_] for id_ in row_indexes]
    else:
        single_class_src_list_np = src_list
        single_class_dst_list_np = dst_list

    class_images_pd_data, class_full_logs = metrics.calculate_image_mAP(single_class_src_list_np,
                                                                        single_class_dst_list_np,
                                                                        metrics.MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                                                        target_class=class_name,
                                                                        iou=iou_threshold, score=score_threshold,
                                                                        need_rez=True)
    projects_pd_data, prj_rez = metrics.calculate_project_mAP(single_class_src_list_np, single_class_dst_list_np,
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
        {"field": "data.perClassSingleImagesTable", "payload": {"columns": g.image_columns,
                                                                "data": class_images_pd_data}},
        {"field": "data.perClassLineChartSeries", "payload": target_chart},
        {"field": "data.perClassSelectedClsTitle", "payload": perClassSelectedClsTitle}
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("view_class")
@sly.timeit
def view_class(api: sly.Api, task_id, context, state, app_logger):
    print('state =', state)
    class_name = state["selectedClassName"]
    iou_threshold = state["IoUThreshold"] / 100
    score_threshold = state['ScoreThreshold'] / 100
    selected_class_metrics(api, task_id, settings.gts, settings.pred, class_name, g.pred_project_info.name,
                           iou_threshold, score_threshold)
    fields = [
        {"field": "state.perClassActiveStep", "payload": 2},
        {"field": "state.perClassActiveNames", "payload": ['per_class_table', 'per_class_image_statistic_table']},
        {"field": "state.perClassShow2", "payload": True},
        {"field": "state.perClassShow3", "payload": False},

        {"field": "data.perClassImageStatTableInfo",
         "payload": "Image Metrics Table and Line Chart for selected class: {}".format(class_name)},
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("show_images_per_class")
@sly.timeit
def show_images_per_class(api: sly.Api, task_id, context, state, app_logger):
    global gallery_per_class
    selected_image_classes = state['selectedClassName']
    if gallery_per_class is None:
        gallery_per_class = CompareGallery(g.task_id, g.api, 'data.perClass', g.aggregated_meta)
    ui_utils.show_images_body(api, task_id, state, gallery_per_class, "data.perClassGalleryTitle",
                              selected_image_classes=selected_image_classes, gallery_table="data.GalleryTable2")
    fields = [
        {"field": "state.perClassActiveStep", "payload": 3},
        {"field": "state.perClassActiveNames",
         "payload": ['per_class_table', 'per_class_image_statistic_table', 'per_class_gallery']},
        {"field": "state.perClassShow2", "payload": True},
        {"field": "state.perClassShow3", "payload": True},
    ]
    api.app.set_fields(task_id, fields)
