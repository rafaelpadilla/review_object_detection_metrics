import supervisely_lib as sly
import numpy as np
import globals as g
from supervisely.src import download_data as dd
from supervisely.src.confusion_matrix import confusion_matrix_py
import metrics
import download_data
from src.bounding_box import BoundingBox, BBType, BBFormat
from src.utils.enumerators import MethodAveragePrecision


def init(data, state):
    state['selection'] = {}
    state['selected'] = None
    conf_matrx_columns_v2 = []
    diagonal_max = 0
    max_value = 0
    cm_data = []

    slyConfusionMatrix = {
        "classes": conf_matrx_columns_v2,
        "diagonalMax": diagonal_max,
        "maxValue": max_value,
        "data": cm_data
    }
    data['slyConfusionMatrix'] = slyConfusionMatrix
    data['confusionTableImages'] = {}
    data['previewContent'] = {}
    data['previewOptions'] = {}


@g.my_app.callback("evaluate_button_click")
@sly.timeit
def evaluate_button_click(api: sly.Api, task_id, context, state, app_logger):
    global cm
    selected_classes = state['selectedClasses']
    percentage = state['samplePercent']
    iou_threshold = state['IoUThreshold']/100
    score_threshold = state['ScoreThreshold']/100
    dd.exec_download_v2(selected_classes, percentage=percentage, confidence_threshold=score_threshold, reload_always=True)
    gt = dd.plt_boxes['gt_images']
    det = dd.plt_boxes['pred_images']
    cm = confusion_matrix_py.confusion_matrix(gt_boxes=gt, det_boxes=det,
                                              iou_threshold=iou_threshold, score_threshold=score_threshold)
    conf_matrx_columns_v2, conf_matrx_data_v2 = confusion_matrix_py.convert_confusion_matrix_to_plt_format_v2(cm)
    diagonal_max = 0
    max_value = 0
    cm_data = list()
    np_table = np.array(conf_matrx_data_v2)
    a0 = np.sum(np_table, axis=0)  # столбец
    a1 = np.sum(np_table, axis=1)  # строка
    a0 = np.append(a0, 0)
    res_table = np.hstack((np_table, a1.reshape(-1, 1)))
    res_table = np.vstack((res_table, a0))

    conf_matrx_data_v2_extended = res_table.tolist()

    for i1, row in enumerate(conf_matrx_data_v2_extended):
        tmp = []
        for i2, col in enumerate(row):
            tmp.append(dict(value=int(col)))
            if i1 == i2 and col > diagonal_max:
                diagonal_max = col
            if i1 != i2 and col > max_value:
                max_value = col
        cm_data.append(tmp)

    slyConfusionMatrix = {
        "data": {
            "classes": conf_matrx_columns_v2,
            "diagonalMax": diagonal_max,
            "maxValue": max_value,
            "data": cm_data
        }
    }
    fields = [
        {"field": "data.slyConfusionMatrix", "payload": slyConfusionMatrix},
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("show_image_table")
@sly.timeit
def show_image_table(api: sly.Api, task_id, context, state, app_logger):
    print('!!!! it works !!!!')
    print('state.selected =', state['selected'])
    print()
    selected_cell = state['selected']
    row_class = selected_cell['rowClass']
    col_class = selected_cell['colClass']

    images_to_show = list(set(cm[col_class][row_class]))
    print('images_to_show =', images_to_show)

    tmp1 = dd.confidence_filtered_data['gt_images']['trainval'][0]
    tmp2 = dd.confidence_filtered_data['pred_images']['trainval'][0]
    print(tmp1)
    print(tmp2)

    selected_image_infos = dict(gt_images=[], pred_images=[])

    for prj_key, prj_value in dd.confidence_filtered_data.items():
        for dataset_key, dataset_value in prj_value.items():
            for element in dataset_value:
                if element['image_name'] in images_to_show:
                    selected_image_infos[prj_key].append(element)

    encoder = BoundingBox
    gts = []
    pred= []
    for gt, pr in zip(selected_image_infos['gt_images'], selected_image_infos['pred_images']):
        assert gt['image_name'] == pr['image_name'], 'different images'
        print('GT =', gt)
        print('PR =', pr)
        gt_boxes = download_data.plt2bb(gt, encoder, bb_type=BBType.GROUND_TRUTH)
        pred_boxes = download_data.plt2bb(pr, encoder, bb_type=BBType.DETECTED)
        gts.append([gt['image_id'], gt['image_name'], gt['full_storage_url'], gt_boxes])
        pred.append([pr['image_id'], pr['image_name'], pr['full_storage_url'], pred_boxes])
    iou_threshold = 0.5
    score_threshold = 0.01
    images_pd_data = metrics.calculate_image_mAP(gts, pred, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                                 iou=iou_threshold, score=score_threshold)
    print('images_pd_data =', images_pd_data)
    fields = [
        {"field": "data.confusionTableImages", "payload": {"columns": metrics.image_columns, "data": images_pd_data}},
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("show_images")
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


cm = {}
