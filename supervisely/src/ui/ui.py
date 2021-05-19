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


def init(data, state):
    input.init(data, state)
    classes.init(data, state)
    datasets.init(data, state)
    settings.init(data, state)
    confusion_matrix.init(data, state)
    metrics.init(data, state)
    pass


@g.my_app.callback("show_image_table")
@sly.timeit
def show_image_table(api: sly.Api, task_id, context, state, app_logger):
    cm = dd.cm
    # print('!!!! it works !!!!')
    # print('state.selected =', state['selected'])
    selected_cell = state['selected']
    row_class = selected_cell['rowClass']
    col_class = selected_cell['colClass']
    iou_threshold = state['IoUThreshold'] / 100
    score_threshold = state['ScoreThreshold'] / 100

    # print('conf matrix =', cm)
    images_to_show = list(set(cm[col_class][row_class]))
    # print('images_to_show =', images_to_show)

    # tmp1 = dd.confidence_filtered_data['gt_images']['trainval'][0]
    # tmp2 = dd.confidence_filtered_data['pred_images']['trainval'][0]
    # print(tmp1)
    # print(tmp2)

    selected_image_infos = dict(gt_images=[], pred_images=[])

    for prj_key, prj_value in dd.filtered_confidences.items():
        for dataset_key, dataset_value in prj_value.items():
            for element in dataset_value:
                if element['image_name'] in images_to_show:
                    selected_image_infos[prj_key].append(element)

    encoder = BoundingBox
    gts = []
    pred = []
    for gt, pr in zip(selected_image_infos['gt_images'], selected_image_infos['pred_images']):
        assert gt['image_name'] == pr['image_name'], 'different images'
        gt_boxes = utils.plt2bb(gt, encoder, bb_type=BBType.GROUND_TRUTH)
        pred_boxes = utils.plt2bb(pr, encoder, bb_type=BBType.DETECTED)
        gts.append([gt['image_id'], gt['image_name'], gt['full_storage_url'], gt_boxes])
        pred.append([pr['image_id'], pr['image_name'], pr['full_storage_url'], pred_boxes])

    images_pd_data = metrics.calculate_image_mAP(gts, pred, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                                 iou=iou_threshold, score=score_threshold)
    fields = [
        {"field": "data.confusionTableImages", "payload": {"columns": metrics.image_columns, "data": images_pd_data}},
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("show_images")
@sly.timeit
def show_images(api: sly.Api, task_id, context, state, app_logger):
    # print('Show_images: ', state)
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
