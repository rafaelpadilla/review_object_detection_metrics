import supervisely_lib as sly
from supervisely.src import download_data as dd
from supervisely.src import utils
from src.bounding_box import BoundingBox, BBType, BBFormat
from src.utils.enumerators import MethodAveragePrecision
# from supervisely_lib.app.widgets.compare_gallery import CompareGallery
# from supervisely_lib.app.widgets.sly_table import SlyTable
from widgets.compare_gallery import CompareGallery
from widgets.sly_table import SlyTable
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

gallery_conf_matrix = CompareGallery(g.task_id, g.api, 'data.CMGallery', g.aggregated_meta)
gallery_per_image = CompareGallery(g.task_id, g.api, 'data.perImage', g.aggregated_meta)
gallery_per_class = CompareGallery(g.task_id, g.api, 'data.perClass', g.aggregated_meta)

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

    text = '''Images for the selected cell in confusion matrix: "{}" (actual) <-> "{}" (predicted)'''.format(row_class,
                                                                                                             col_class)

    if col_class != 'None' and row_class != 'None':
        if len(list(cm[col_class][row_class])) == 1:
            description_1 = '''{} "{}" objects is detected as "{}"'''.format(len(list(cm[col_class][row_class])),
                                                                             row_class, col_class)
        else:
            description_1 = '''{} "{}" objects are detected as "{}"'''.format(len(list(cm[col_class][row_class])),
                                                                              row_class, col_class)
    else:
        description_1 = None

    if len(list(cm['None'][row_class])) != 0:
        if len(list(cm['None'][row_class])) == 1:
            description_2 = '''{} "{}" object is not detected"'''.format(len(list(cm['None'][row_class])),
                                                                         row_class)
        else:
            description_2 = '''{} "{}" objects are not detected"'''.format(len(list(cm['None'][row_class])),
                                                                           row_class)
    else:
        description_2 = None

    if len(list(cm[col_class]['None'])) != 0:
        if len(list(cm[col_class]['None'])) == 1:
            description_3 = '''Model predicted {} "{}" object that is not in GT (None <-> {})"'''.format(
                len(list(cm[col_class]['None'])), col_class, col_class)
        else:
            description_3 = '''Model predicted {} "{}" objects that are not in GT (None <-> {})'''.format(
                len(list(cm[col_class]['None'])), col_class, col_class)
    else:
        description_3 = None

    fields = [
        {"field": "data.CMImageTableTitle", "payload": text},
        {"field": "data.CMImageTableDescription1", "payload": description_1},
        {"field": "data.CMImageTableDescription2", "payload": description_2},
        {"field": "data.CMImageTableDescription3", "payload": description_3}
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
def filter_classes(ann, selected_classes, score=None):
    ann = sly.Annotation.from_json(ann.annotation, g.aggregated_meta)
    tmp_list = list()
    tag_list = list()
    for ii in ann.labels:
        if ii.obj_class.name in selected_classes:
            # print('ii.obj_class.name =', ii.obj_class.name)
            tmp_list.append(ii)
            if score is not None:
                # print('ii.tags =', ii.tags)
                for ij in ii.tags:
                    if ij.value >= score:
                        # print(ij)
                        # print(ij.value)
                        tag_list.append(ij)

    ann = ann.clone(labels=tmp_list)
    if score is not None:
        ann = ann.clone(img_tags=tag_list)
    return ann


def show_images_body(api, task_id, state, gallery_template, v_model, selected_image_classes=None):
    # print('state =', state)
    selected_classes = state['selectedClasses'] if selected_image_classes is None else selected_image_classes

    selected_row_data = state["selection"]["selectedRowData"]

    try:
        image_name = selected_row_data['name'].split('_blank">')[-1].split('</')[0]
    except:
        image_name = 'empty state'

    score = state['ScoreThreshold'] / 100
    # print('scoreThreshold =', score)
    if selected_row_data is not None and state["selection"]["selectedColumnName"] is not None:
        keys = [key for key in selected_row_data]
        if 'SRC_ID' not in keys:
            return
        image_id_1 = int(selected_row_data['SRC_ID'])
        image_id_2 = int(selected_row_data['DST_ID'])
    else:
        return

    ann_1 = filter_classes(api.annotation.download(image_id_1), selected_classes)
    ann_2 = filter_classes(api.annotation.download(image_id_2), selected_classes, score)

    gallery_template.set_left(title='original', ann=ann_1,
                              image_url=api.image.get_info_by_id(image_id_1).full_storage_url)
    gallery_template.set_right(title='detection', ann=ann_2,
                               image_url=api.image.get_info_by_id(image_id_2).full_storage_url)
    gallery_template.update()

    text = 'Grid gallery for {}'.format(image_name)
    fields = [
        {"field": v_model, "payload": text},
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("show_images_confusion_matrix")
@sly.timeit
def show_images_confusion_matrix(api: sly.Api, task_id, context, state, app_logger):
    show_images_body(api, task_id, state, gallery_conf_matrix, "data.CMGalleryTitle")


@g.my_app.callback("show_images_per_image")
@sly.timeit
def show_images_per_image(api: sly.Api, task_id, context, state, app_logger):
    show_images_body(api, task_id, state, gallery_per_image, "data.perImageGalleryTitle")


@g.my_app.callback("show_images_per_class")
@sly.timeit
def show_images_per_class(api: sly.Api, task_id, context, state, app_logger):
    selected_image_classes = state['selectedClassName']
    show_images_body(api, task_id, state, gallery_per_class, "data.perClassGalleryTitle", selected_image_classes)
