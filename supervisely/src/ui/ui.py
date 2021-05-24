import supervisely_lib as sly
from supervisely.src import download_data as dd
from supervisely.src import utils
from src.bounding_box import BoundingBox, BBType, BBFormat
from src.utils.enumerators import MethodAveragePrecision
from supervisely_lib.app.widgets.compare_gallery import CompareGallery
from supervisely_lib.app.widgets.sly_table import SlyTable

import input
import classes
import settings
import datasets
import confusion_matrix
import metrics
import per_image_metrics
import per_class_metrics
import overall_metrics
import globals as g

aggregated_meta = {'classes': [], 'tags': g._pred_meta_['tags'], 'projectType': 'images'}
for i in g._gt_meta_['classes']:
    for j in g._pred_meta_['classes']:
        if i['title'] == j['title'] and i['shape'] == j['shape']:
            aggregated_meta['classes'].append(i)

aggregated_meta = sly.ProjectMeta.from_json(aggregated_meta)
gallery_conf_matrix = CompareGallery(g.task_id, g.api, 'data.CMGallery', aggregated_meta)
gallery_per_image = CompareGallery(g.task_id, g.api, 'data.perImage', aggregated_meta)
gallery_per_class = CompareGallery(g.task_id, g.api, 'data.perClass', aggregated_meta)

cm_image_table = SlyTable(g.api, g.task_id, 'data.CMTableImages', metrics.image_columns)


def init(data, state):
    input.init(data, state)
    classes.init(data, state)
    datasets.init(data, state)
    settings.init(data, state)
    confusion_matrix.init(data, state)
    per_image_metrics.init(data, state)
    per_class_metrics.init(data, state)
    overall_metrics.init(data, state)


def show_image_table_body(api, task_id, state, v_model):
    print('state =', state)
    selected_cell = state['selected']
    row_class = selected_cell['rowClass']
    col_class = selected_cell['colClass']
    iou_threshold = state['IoUThreshold'] / 100
    score_threshold = state['ScoreThreshold'] / 100

    cm = dd.cm
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

    text = '''Images for the selected cell in confusion matrix: "{}" (actual) <-> "{}" (predicted)'''.format(row_class, col_class)

    if len(list(cm[col_class][row_class])) == 1:
        description_1 = '''{} "{}" objects is detected as "{}"'''.format(len(list(cm[col_class][row_class])),
                                                                          row_class, col_class)
    else:
        description_1 = '''{} "{}" objects are detected as "{}"'''.format(len(list(cm[col_class][row_class])),
                                                                          row_class, col_class)
    fields = [
        {"field": "data.CMImageTableTitle", "payload": text},
        {"field": "data.CMImageTableDescription", "payload": description_1}
    ]
    api.app.set_fields(task_id, fields)

    cm_image_table.set_data(images_pd_data)
    cm_image_table.update()


@g.my_app.callback("show_image_table_cm")
@sly.timeit
def show_image_table_cm(api: sly.Api, task_id, context, state, app_logger):
    v_model = "data.CMTableImages"
    show_image_table_body(api, task_id, state, v_model)


# ======================================================================================================================
def filter_classes(ann, selected_classes):
    ann = sly.Annotation.from_json(ann.annotation, aggregated_meta)
    tmp_list = list()
    for ii in ann.labels:
        if ii.obj_class.name in selected_classes:
            tmp_list.append(ii)
    return sly.Annotation(ann.img_size, tmp_list, ann.img_tags, ann.img_description)


def show_images_body(api, state, gallery_template):
    selected_classes = state['selectedClasses']
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

    ann_1 = filter_classes(api.annotation.download(image_id_1), selected_classes)
    ann_2 = filter_classes(api.annotation.download(image_id_2), selected_classes)
    gallery_template.set_left(title='original', ann=ann_1,
                              image_url=api.image.get_info_by_id(image_id_1).full_storage_url)
    gallery_template.set_right(title='detection', ann=ann_2,
                               image_url=api.image.get_info_by_id(image_id_2).full_storage_url)
    gallery_template.update()


@g.my_app.callback("show_images_confusion_matrix")
@sly.timeit
def show_images_confusion_matrix(api: sly.Api, task_id, context, state, app_logger):
    show_images_body(api, state, gallery_conf_matrix)


@g.my_app.callback("show_images_per_image")
@sly.timeit
def show_images_per_image(api: sly.Api, task_id, context, state, app_logger):
    show_images_body(api, state, gallery_per_image)


@g.my_app.callback("show_images_per_class")
@sly.timeit
def show_images_per_class(api: sly.Api, task_id, context, state, app_logger):
    show_images_body(api, state, gallery_per_class)
