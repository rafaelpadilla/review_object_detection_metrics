import supervisely_lib as sly
import globals as g
from supervisely.src import download_data as dd
from supervisely.src import utils
from src.bounding_box import BoundingBox, BBType, BBFormat
from src.utils.enumerators import MethodAveragePrecision

import input
import classes
import settings
import datasets
import confusion_matrix
import metrics
import per_image_metrics
import per_class_metrics
import overall_metrics


def init(data, state):
    input.init(data, state)
    classes.init(data, state)
    datasets.init(data, state)
    settings.init(data, state)
    confusion_matrix.init(data, state)
    per_image_metrics.init(data, state)
    per_class_metrics.init(data, state)
    overall_metrics.init(data, state)


@g.my_app.callback("show_image_table")
@sly.timeit
def show_image_table(api: sly.Api, task_id, context, state, app_logger):
    cm = dd.cm
    # print('!!!! it works !!!!')
    print('state =', state)
    selected_cell = state['selected']
    row_class = selected_cell['rowClass']
    col_class = selected_cell['colClass']
    iou_threshold = state['IoUThreshold'] / 100
    score_threshold = state['ScoreThreshold'] / 100

    images_to_show = list(set(cm[col_class][row_class]))

    selected_image_infos = dict(gt_images=[], pred_images=[])

    for prj_key, prj_value in dd.filtered_confidences.items():
        for dataset_key, dataset_value in prj_value.items():
            for element in dataset_value:
                if element['image_name'] in images_to_show:
                    selected_image_infos[prj_key].append(element)

    encoder = BoundingBox
    gts = []
    pred = []
    assert len(selected_image_infos['gt_images']) == len(selected_image_infos['pred_images'])
    dataset_names = {}
    for gt, pr in zip(selected_image_infos['gt_images'], selected_image_infos['pred_images']):
        assert gt['image_name'] == pr['image_name'], 'different images'
        gt_boxes = utils.plt2bb(gt, encoder, bb_type=BBType.GROUND_TRUTH)
        pred_boxes = utils.plt2bb(pr, encoder, bb_type=BBType.DETECTED)

        if gt['dataset_id'] not in dataset_names:
            dataset_names[gt['dataset_id']] = api.dataset.get_info_by_id(gt['dataset_id']).name
        if pr['dataset_id'] not in dataset_names:
            dataset_names[pr['dataset_id']] = api.dataset.get_info_by_id(pr['dataset_id']).name

        gts.append([gt['image_id'], gt['image_name'], gt['full_storage_url'],
                    dataset_names[gt['dataset_id']], gt_boxes])
        pred.append([pr['image_id'], pr['image_name'], pr['full_storage_url'],
                     dataset_names[pr['dataset_id']], pred_boxes])

    images_pd_data = metrics.calculate_image_mAP(gts, pred, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                                 iou=iou_threshold, score=score_threshold)
    fields = [
        {"field": "data.confusionTableImages", "payload": {"columns": metrics.image_columns, "data": images_pd_data}},
    ]
    api.app.set_fields(task_id, fields)


def show_images_body(api, state):
    selected_lasses = state['selectedClasses']
    selected_row_data = state["selection"]["selectedRowData"]
    score = state['ScoreThreshold']/100
    if selected_row_data is not None and state["selection"]["selectedColumnName"] is not None:
        keys = [key for key in selected_row_data]
        if 'SRC_ID' not in keys:
            return
        image_id_1 = int(selected_row_data['SRC_ID'])
        image_id_2 = int(selected_row_data['DST_ID'])
    else:
        return

    ann_1 = api.annotation.download(image_id_1)
    ann_2 = api.annotation.download(image_id_2)

    avalible_objects = []
    for object_ in ann_1.annotation['objects']:
        if object_['classTitle'] in selected_lasses:
            avalible_objects.append(object_)
    ann_1.annotation['objects'] = avalible_objects

    avalible_objects = []
    for object_ in ann_2.annotation['objects']:
        if object_['classTitle'] in selected_lasses:
            # print("object_['tags'][0]['value'] =", object_['tags'][0]['value'])
            if object_['tags'][0]['value'] >= score:
                avalible_objects.append(object_)
    ann_2.annotation['objects'] = avalible_objects

    content = {
        "projectMeta": g.gt_meta.to_json(),
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
    return content


@g.my_app.callback("show_images_confusion_matrix")
@sly.timeit
def show_images_confusion_matrix(api: sly.Api, task_id, context, state, app_logger):
    # print('Show_images: ', state)
    content = show_images_body(api, state)

    fields = [
        {"field": "data.confusionMatrixPreviewContent", "payload": content},
        {"field": "data.confusionMatrixPreviewOptions", "payload": options},
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("show_images_per_image")
@sly.timeit
def show_images_per_image(api: sly.Api, task_id, context, state, app_logger):
    # print('Show_images: ', state)
    content = show_images_body(api, state)

    fields = [
        {"field": "data.perImagesPreviewContent", "payload": content},
        {"field": "data.perImagesPreviewOptions", "payload": options},
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("show_images_per_class")
@sly.timeit
def show_images_per_class(api: sly.Api, task_id, context, state, app_logger):
    # print('Show_images: ', state)
    content = show_images_body(api, state)
    fields = [
        {"field": "data.perClassPreviewContent", "payload": content},
        {"field": "data.perClassPreviewOptions", "payload": options},
    ]
    api.app.set_fields(task_id, fields)


options = {
    "showOpacityInHeader": False,
    "opacity": 0.8,
    "fillRectangle": False,
    "syncViews": True,
    "syncViewsBindings": [["ann_1", "ann_2"]]
}
