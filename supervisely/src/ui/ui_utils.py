from supervisely.src import download_data as dd
from src.bounding_box import BoundingBox, BBType, BBFormat
from supervisely.src import utils
from src.utils.enumerators import MethodAveragePrecision
import metrics
import supervisely_lib as sly
import globals as g
import settings


def show_image_table_body(api, task_id, state, v_model, image_table):
    print('state =', state)
    selected_cell = state['selected']
    row_class = selected_cell['rowClass']
    col_class = selected_cell['colClass']
    iou_threshold = state['IoUThreshold'] / 100
    score_threshold = state['ScoreThreshold'] / 100

    # cm = dd.cm
    cm = settings.cm
    images_to_show = list(set(cm[col_class][row_class]))

    selected_image_infos = dict(gt_images=[], pred_images=[])

    for prj_key, prj_value in settings.filtered_confidences.items():
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
        {"field": "data.CMImageTableDescription3", "payload": description_3},
    ]
    api.app.set_fields(task_id, fields)

    image_table.set_data(images_pd_data)
    image_table.update()


def filter_classes(ann, selected_classes, score=None):
    ann = sly.Annotation.from_json(ann.annotation, g.aggregated_meta)
    tmp_list = list()
    tag_list = list()
    for ii in ann.labels:
        if ii.obj_class.name in selected_classes:
            tmp_list.append(ii)
            if score is not None:
                for ij in ii.tags:
                    if ij.value >= score:
                        tag_list.append(ij)

    ann = ann.clone(labels=tmp_list)
    if score is not None:
        ann = ann.clone(img_tags=tag_list)
    return ann


def show_images_body(api, task_id, state, gallery_template, v_model, selected_image_classes=None):
    selected_classes = state['selectedClasses'] if selected_image_classes is None else selected_image_classes
    selected_row_data = state["selection"]["selectedRowData"]

    try:
        image_name = selected_row_data['name'].split('_blank">')[-1].split('</')[0]
    except:
        image_name = 'empty state'

    score = state['ScoreThreshold'] / 100
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

    text = 'Gallery for {}'.format(image_name)
    fields = [
        {"field": v_model, "payload": text},
    ]
    api.app.set_fields(task_id, fields)
