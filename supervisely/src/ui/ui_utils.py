import numpy as np

import pandas as pd
from src.utils.enumerators import MethodAveragePrecision
from src.evaluators.pascal_voc_evaluator import calculate_ap_every_point
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

    cm = settings.cm
    images_to_show = list(set(cm[col_class][row_class]))
    dataset_names = {}
    pred_images_names_list = []

    prepared_data = {'gt_images': settings.gts, 'pred_images': settings.pred}
    selected_image_infos = {}
    for prj_key, prj_value in prepared_data.items():
        selected_image_infos.setdefault(prj_key, {})
        for dataset_key, dataset_value in prj_value.items():
            selected_image_infos[prj_key].setdefault(dataset_key, [])
            for element in dataset_value:
                if element[1] in images_to_show:
                    selected_image_infos[prj_key][dataset_key].append(element)
                    if prj_key == 'pred_images':
                        pred_images_names_list.append(element[1])

    assert len(selected_image_infos['gt_images']) == len(selected_image_infos['pred_images'])
    gts, pred = selected_image_infos['gt_images'], selected_image_infos['pred_images']
    projects_pd_data, prj_rez = metrics.calculate_project_mAP(src_list=settings.gts,
                                                    dst_list=settings.pred,
                                                    method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                                    dst_project_name=g.pred_project_info.name,
                                                    iou=iou_threshold, score=score_threshold)

    list_rez = []
    for class_ in prj_rez['per_class']:
        prj_rez['per_class'][class_]['table']['class'] = class_
        list_rez.append(prj_rez['per_class'][class_]['table'])
    list_rez = pd.concat(list_rez)

    columns = ['TP', 'FP']
    agg_df = list_rez.groupby(['image']).sum()[columns].reset_index()

    gt_npos = {}
    gt_cls_obj_num = {}
    for dataset_name, dataset_values in settings.gts.items():
        for image in dataset_values:
            gt_npos[image[1]] = len(image[-1])
            gt_cls_obj_num[image[1]] = {}
            for bbox in image[-1]:
                cls_id = bbox.get_class_id()
                if cls_id in gt_cls_obj_num[image[1]]:
                    gt_cls_obj_num[image[1]][cls_id] += 1
                else:
                    gt_cls_obj_num[image[1]][cls_id] = 1

    agg_df['NPOS'] = agg_df.image.apply(lambda x: np.float(gt_npos[x]))

    def set_column_v2(data):
        tp = data[1]
        fp = data[2]
        precision = tp / (tp + fp)
        npos = data[3]
        recall = tp / npos  # (TP + FN)
        return round(precision, 2), round(recall, 2)

    agg_df['precision_recall'] = agg_df.apply(set_column_v2, axis=1)

    ret = {}
    image_map = {}
    for image in agg_df['image']:
        lines = list_rez.loc[list_rez['image'] == image]
        cls_in_image = set(lines['class'])
        for cls in cls_in_image:
            cls_lines = lines.loc[lines['class'] == cls]

            acc_FP = np.cumsum(cls_lines['FP'].to_list())
            acc_TP = np.cumsum(cls_lines['TP'].to_list())
            try:
                rec = acc_TP / gt_cls_obj_num[image][cls]
            except:
                rec = np.zeros_like(acc_TP)
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            [ap, _, _, _] = calculate_ap_every_point(rec, prec)
            ret[cls] = {
                'AP': ap
            }
        gt_classes_only = gt_cls_obj_num[image].keys()
        image_map[image] = sum([v['AP'] for k, v in ret.items() if k in gt_classes_only]) / len(gt_classes_only)

    def set_map(img_name):
        return round(image_map[img_name], 2)

    agg_df['mAP'] = agg_df['image'].apply(set_map)

    images_pd_data = metrics.calculate_image_mAP(gts, pred, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                                 iou=iou_threshold, score=score_threshold)

    for image in images_pd_data:
        name = image[3].split('_blank">')[-1].split('</a>')[0]
        extra_data = agg_df.loc[agg_df['image'] == name]
        # print(image)
        image[4] = extra_data['TP'].values[0]
        image[5] = extra_data['FP'].values[0]
        image[6] = extra_data['NPOS'].values[0]

        image[7] = extra_data['precision_recall'].to_list()[0][0]
        image[8] = extra_data['precision_recall'].to_list()[0][1]
        image[9] = extra_data['mAP'].values[0]

    text = '''Images for the selected cell in confusion matrix: "{}" (ground truth) <-> "{}" (prediction)'''.format(row_class, col_class)

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
            description_3 = '''Model predicted {} "{}" object that is not in ground truth (None <-> {})"'''.format(
                len(list(cm[col_class]['None'])), col_class, col_class)
        else:
            description_3 = '''Model predicted {} "{}" objects that are not in ground truth (None <-> {})'''.format(
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


def show_images_body(api, task_id, state, gallery_template, v_model, gallery_table, selected_image_classes=None):
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

    # ann_1 = filter_classes(api.annotation.download(image_id_1), selected_classes)
    # ann_2 = filter_classes(api.annotation.download(image_id_2), selected_classes, score)

    ann_1 = sly.Annotation.from_json(api.annotation.download(image_id_1).annotation, g.aggregated_meta)
    ann_2 = sly.Annotation.from_json(api.annotation.download(image_id_2).annotation, g.aggregated_meta)

    dataset_name = selected_row_data['dataset_name']
    image_map_ = settings.object_mapper[dataset_name][image_name]

    dict_ = []
    ddict_ = []
    data = []
    for key in image_map_:
        data.append(image_map_[key])
    data_np = np.asarray(data).transpose()

    for line in data_np:
        gt = None
        l_name = None
        l_color = None
        pr = None
        r_name = None
        r_color = None

        for l in ann_1.labels:
            try:
                if int(line[0]) == l.geometry.sly_id:
                    gt = int(line[0]) if int(line[0]) is not None else None
                    l_name = l.obj_class.name
                    l_color = l.obj_class.to_json()['color']
                    break
            except:
                gt = None
                l_name = None
                l_color = None

        for r in ann_2.labels:

            try:
                if int(line[1]) == r.geometry.sly_id:
                    pr = int(line[1]) if int(line[1]) is not None else None
                    r_name = r.obj_class.name
                    r_color = r.obj_class.to_json()['color']
                    break
            except:
                pr = None
                r_name = None
                r_color = None
        dd = {
            "gt": {"id": gt, "class": l_name, "color": l_color},
            "pr": {"id": pr, "class": r_name, "color": r_color},
            "mark": line[2],
            "iou": round(float(line[4]), 3) if line[4] is not None else None,
            "conf": round(float(line[3]), 3) if line[3] is not None else None,
            "id_pair": [gt, pr]
        }

        # d = {
        #     "GroundTruth": int(line[0]),
        #     "IoU": line[4],
        #     "Mark": line[2],
        #     "Confidence": line[3],
        #     "Prediction": int(line[1])
        # }
        # dict_.append(d)
        ddict_.append(dd)

    gallery_template.set_left(title='ground truth', ann=ann_1,
                              image_url=api.image.get_info_by_id(image_id_1).full_storage_url)
    gallery_template.set_right(title='predictions', ann=ann_2,
                               image_url=api.image.get_info_by_id(image_id_2).full_storage_url)
    gallery_template.update()

    text = 'Labels preview for {}'.format(image_name)
    fields = [
        {"field": v_model, "payload": text},
        {"field": 'data.GalleryTable', "payload": dict_},
        {"field": gallery_table, "payload": ddict_},
    ]
    api.app.set_fields(task_id, fields)
