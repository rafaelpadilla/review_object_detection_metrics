import supervisely_lib as sly
import globals as g
import ui
import numpy as np
import random
import shelve
from src.bounding_box import BoundingBox as RepoBoundingBox
from supervisely.src.confusion_matrix.bounding_box_py import CoordinatesType, BBType, BBFormat, BoundingBox
from supervisely.src.ui.confusion_matrix import calculate_confusion_matrix
import utils
from supervisely.src.ui import metrics, overall_metrics, per_image_metrics, per_class_metrics


def get_intersected_datasets(img_dict, show_info=False):
    gt_images = set(list(img_dict['gt_images']))
    pred_images = set(list(img_dict['pred_images']))
    intersected_datasets = list(gt_images & pred_images)
    if show_info:
        print('gt_images datasets =', gt_images)
        print('pred_images datasets =', pred_images)
        print('intersected datasets =', intersected_datasets)
    return intersected_datasets


def get_sample_size(dataset_length, percentage):
    sample_size = int(np.ceil(dataset_length / 100 * percentage))
    return sample_size


def get_random_sample(image_dict, intersected_datasets, percentage):
    global total_image_num
    indexes = {}
    for key, value in image_dict['gt_images'].items():
        if key not in intersected_datasets:
            continue
        dataset_length = len(value)
        total_image_num[key] = dataset_length
        sample_size = get_sample_size(dataset_length, percentage)
        indexes[key] = random.sample(range(dataset_length), sample_size)
    return indexes


def download(image_dict, percentage, batch_size=10, show_info=False):
    sample_ = dict()
    intersected_datasets = get_intersected_datasets(image_dict)
    if show_info:
        print('intersected_datasets =', intersected_datasets)
        print('image_dict.keys() =', image_dict.keys())

    indexes = get_random_sample(image_dict, intersected_datasets, percentage)
    for project_key, project_info in image_dict.items():  # project_key in []
        if show_info:
            print('project_key =', project_key)
        sample_[project_key] = {}
        for dataset_key, dataset_info in project_info.items():
            if show_info:
                print('dataset_key =', dataset_key)
            if dataset_key not in intersected_datasets:
                continue
            sample_[project_key][dataset_key] = []
            dataset_id = dataset_info[0].dataset_id
            slice_to_download = [dataset_info[index] for index in indexes[dataset_key]]
            for ix, batch in enumerate(sly.batched(slice_to_download, batch_size)):
                image_ids = [image_info.id for image_info in batch]
                annotations = g.api.annotation.download_batch(dataset_id, image_ids)
                sample_[project_key][dataset_key].extend(annotations)
        if show_info:
            print('sample_[', project_key, '] =', len(sample_[project_key]))
    return sample_


def filter_classes(image_list, classes_names):
    new_image_list = []
    for image in image_list:
        bboxes = image['annotation']['objects']  # .annotation
        new_box_list = []
        for bbox in bboxes:
            if bbox['classTitle'] in classes_names:
                new_box_list.append(bbox)
        if new_box_list:
            image['annotation']['objects'] = new_box_list
            new_image_list.append(image)
    return new_image_list


def filter_confidences(image_list, confidence_threshold):
    new_image_list = []
    for image in image_list:
        bboxes = image['annotation']['objects']
        # print('filter_confidences: image =', image)
        # print('bboxes =', bboxes)
        if len(bboxes) < 1:
            continue
        new_box_list = []
        for bbox in bboxes:
            if bbox['tags'][0]['value'] > confidence_threshold:
                new_box_list.append(bbox)
        image['annotation']['objects'] = new_box_list
        new_image_list.append(image)
    return new_image_list


# _sample_ = dict()
# def exec_download(classes_names, percentage, confidence_threshold, reload_always=False):
#     global _sample_, plt_boxes, previous_percentage
#     current_dataset = {}
#     if reload_always:
#         _sample_ = download(image_dict=ui.datasets.image_dict, percentage=percentage)
#         current_dataset = _sample_.copy()
#     else:
#         if percentage!=0 and previous_percentage < percentage:
#             previous_percentage = percentage
#             _sample_ = download(image_dict=ui.datasets.image_dict, percentage=percentage)
#             current_dataset = _sample_.copy()
#         else:
#             intersected_datasets = list(current_dataset.keys())
#             current_sample_sizes = dict()
#             for k, v in _sample_['gt_images'].items():
#                 current_sample_sizes[k] = get_sample_size(dataset_length=len(v), percentage=percentage)
#             current_dataset = dict()
#             random_indexes = get_random_sample(_sample_, intersected_datasets, percentage,
#                                                __sample=current_sample_sizes)
#             for prj_key, prj_value in _sample_.items():
#                 current_dataset[prj_key] = dict()
#                 for dataset_key, dataset_value in prj_value.items():
#                     current_dataset[prj_key][dataset_key] = [_sample_[prj_key][dataset_key][index]
#                                                              for index in random_indexes[dataset_key]]
#
#     filtered_classes = {}
#     for prj_key, prj_value in current_dataset.items():  # gt + pred
#         filtered_classes[prj_key] = {}
#         for dataset_key, dataset_value in prj_value.items():
#             filtered_classes[prj_key][dataset_key] = filter_classes(dataset_value, classes_names)
#
#     confidence_filtered_data = {}
#
#     for prj_key, prj_value in filtered_classes.items():  # gt + pred
#         confidence_filtered_data[prj_key] = {}
#         for dataset_key, dataset_value in prj_value.items():
#             if prj_key == 'gt_images':
#                 confidence_filtered_data[prj_key][dataset_key] = dataset_value
#             if prj_key == 'pred_images':
#                 confidence_filtered_data[prj_key][dataset_key] = filter_confidences(dataset_value, confidence_threshold)
#
#     plt_boxes = {}
#     for prj_key, prj_value in filtered_classes.items():
#         plt_boxes[prj_key] = []
#         bb_type = BBType.GROUND_TRUTH if prj_key == 'gt_images' else BBType.DETECTED
#         for dataset_key, dataset_value in prj_value.items():
#             for element in dataset_value:
#                 boxes = plt2bb(batch_element=element, encoder=BoundingBox, bb_type=bb_type)
#                 plt_boxes[prj_key].extend(boxes)


def get_key_by_value(dictionary, target_value):
    idx_list = [i for i, val in enumerate(list(dictionary.values())) if val == target_value]
    key_list = [list(dictionary.keys())[idx] for idx in idx_list]
    return key_list


def download_v2(image_dict, percentage, cache, batch_size=10, show_info=False):
    intersected_datasets = get_intersected_datasets(image_dict)
    if show_info:
        print('intersected_datasets =', intersected_datasets)
        print('image_dict.keys() =', image_dict.keys())
    sample = {}
    indexes = get_random_sample(image_dict, intersected_datasets, percentage)
    for project_key, project_info in image_dict.items():  # project_key in []
        if show_info:
            print('project_key =', project_key)
        sample[project_key] = dict()
        for dataset_key, dataset_info in project_info.items():
            if show_info:
                print('dataset_key =', dataset_key)
            if dataset_key not in intersected_datasets:
                continue
            sample[project_key][dataset_key] = list()
            dataset_id = dataset_info[0].dataset_id
            sample[project_key][dataset_key] = [cache[str(dataset_info[index].id)]
                         for index in indexes[dataset_key] if str(dataset_info[index].id) in cache]
            to_download_indexes = [index for index in indexes[dataset_key] if str(dataset_info[index].id) not in cache]
            slice_to_download = [dataset_info[index] for index in to_download_indexes]
            for ix, batch in enumerate(sly.batched(slice_to_download, batch_size)):
                image_ids = [image_info.id for image_info in batch]
                annotations = g.api.annotation.download_batch(dataset_id, image_ids)
                for batch_, annotation in zip(batch, annotations):
                    batch_image_id = batch_.id
                    annotation_image_id = annotation.image_id
                    assert batch_image_id == annotation_image_id, 'different images'
                    # print('batch_ =', batch_)
                    dict_ = dict(image_id=batch_.id, image_name=batch_.name, labels_count=batch_.labels_count,
                                 dataset_id=batch_.dataset_id, annotation=annotation.annotation,
                                 full_storage_url=batch_.full_storage_url)
                    cache[str(batch_image_id)] = dict_
                    sample[project_key][dataset_key].append(dict_)
            image_names = [d['image_name'] for d in sample[project_key][dataset_key]]
            indexes_to_sort = np.argsort(image_names)
            sample[project_key][dataset_key] = [sample[project_key][dataset_key][index] for index in indexes_to_sort]
    return sample


def class_filtering(dataset, classes_names):
    filtered_class = {}
    for prj_key, prj_value in dataset.items():  # gt + pred
        filtered_class[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            filtered_class[prj_key][dataset_key] = filter_classes(dataset_value, classes_names)

    for dataset_key in dataset[list(dataset.keys())[0]]:
        gt_image_names = [image['image_name'] for image in filtered_class['gt_images'][dataset_key]]
        pred_image_names = [image['image_name'] for image in filtered_class['pred_images'][dataset_key]]
        # print('gt_image_names =', gt_image_names)
        # print('pred_image_names =', pred_image_names)
        list1 = gt_image_names
        list2 = pred_image_names
        only_in_first  = [item for item in list1 if item not in list2]
        only_in_second = [item for item in list2 if item not in list1]
        # print('names only_in_gt_images =', only_in_first)
        # print('names only_in_pred_images =', only_in_second)

        additional_gt_images = [image_info for image_info in dataset['gt_images'][dataset_key] if
                                image_info['image_name'] in only_in_second]
        new_image_list = []
        for image in additional_gt_images:
            bboxes = image['annotation']['objects']  # .annotation
            new_box_list = []
            for bbox in bboxes:
                if bbox['classTitle'] in classes_names:
                    new_box_list.append(bbox)
            image['annotation']['objects'] = new_box_list
            new_image_list.append(image)
        additional_gt_images = new_image_list

        additional_pred_images = [image_info for image_info in dataset['pred_images'][dataset_key] if
                                  image_info['image_name'] in only_in_first]
        new_image_list = []
        for image in additional_pred_images:
            bboxes = image['annotation']['objects']  # .annotation
            new_box_list = []
            for bbox in bboxes:
                if bbox['classTitle'] in classes_names:
                    new_box_list.append(bbox)
            image['annotation']['objects'] = new_box_list
            new_image_list.append(image)
        additional_pred_images = new_image_list

        filtered_class['gt_images'][dataset_key].extend(additional_gt_images)
        filtered_class['pred_images'][dataset_key].extend(additional_pred_images)
        gt_image_names_to_sort = [image_info['image_name'] for image_info in filtered_class['gt_images'][dataset_key]]
        pred_image_names_to_sort = [image_info['image_name'] for image_info in filtered_class['pred_images'][dataset_key]]
        indexes_gt = np.argsort(gt_image_names_to_sort)
        indexes_pred = np.argsort(pred_image_names_to_sort)
        filtered_class['gt_images'][dataset_key] = [filtered_class['gt_images'][dataset_key][i] for i in indexes_gt]
        filtered_class['pred_images'][dataset_key] = [filtered_class['pred_images'][dataset_key][i] for i in indexes_pred]
        # print('after addition')
        # for i, j in zip(filtered_class['gt_images'][dataset_key], filtered_class['pred_images'][dataset_key]):
        #     print(i['image_name'], j['image_name'], i['image_name'] == j['image_name'])
    return filtered_class


def confidence_filtering(dataset, confidence_threshold):
    filtered_confidence = {}
    for prj_key, prj_value in dataset.items():  # gt + pred
        filtered_confidence[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            if prj_key == 'gt_images':
                filtered_confidence[prj_key][dataset_key] = dataset_value
            if prj_key == 'pred_images':
                filtered_confidence[prj_key][dataset_key] = filter_confidences(dataset_value, confidence_threshold)

    for dataset_key in dataset[list(dataset.keys())[0]]:
        gt_image_names = [image['image_name'] for image in filtered_confidence['gt_images'][dataset_key]]
        pred_image_names = [image['image_name'] for image in filtered_confidence['pred_images'][dataset_key]]
        # print('gt_image_names =', gt_image_names)
        # print('pred_image_names =', pred_image_names)
        list1 = gt_image_names
        list2 = pred_image_names
        only_in_first = [item for item in list1 if item not in list2]
        only_in_second = [item for item in list2 if item not in list1]
        # print('names only_in_gt_images =', only_in_first)
        # print('names only_in_pred_images =', only_in_second)
        additional_gt_images = [image_info for image_info in dataset['gt_images'][dataset_key] if
                                image_info['image_name'] in only_in_second]
        additional_pred_images = [image_info for image_info in dataset['pred_images'][dataset_key] if
                                  image_info['image_name'] in only_in_first]
        new_image_list = []
        for image in additional_pred_images:
            bboxes = image['annotation']['objects']
            # print('filter confidences additional: bbox =', bboxes)
            if bboxes:
                new_box_list = []
                for bbox in bboxes:

                    if bbox['tags'][0]['value'] > confidence_threshold:
                        new_box_list.append(bbox)
                image['annotation']['objects'] = new_box_list
            new_image_list.append(image)
        additional_pred_images = new_image_list

        filtered_confidence['gt_images'][dataset_key].extend(additional_gt_images)
        filtered_confidence['pred_images'][dataset_key].extend(additional_pred_images)

        gt_image_names_to_sort = [image_info['image_name'] for image_info in
                                  filtered_confidence['gt_images'][dataset_key]]
        pred_image_names_to_sort = [image_info['image_name'] for image_info in
                                    filtered_confidence['pred_images'][dataset_key]]
        indexes_gt = np.argsort(gt_image_names_to_sort)
        indexes_pred = np.argsort(pred_image_names_to_sort)
        filtered_confidence['gt_images'][dataset_key] = [filtered_confidence['gt_images'][dataset_key][i]
                                                         for i in indexes_gt]
        filtered_confidence['pred_images'][dataset_key] = [filtered_confidence['pred_images'][dataset_key][i]
                                                           for i in indexes_pred]
    return filtered_confidence


def exec_download_v2(classes_names, percentage, confidence_threshold, show_logs=False):
    global current_dataset, filtered_classes, filtered_confidences, plt_boxes
    db = shelve.open(filename='db', writeback=True)
    current_dataset = download_v2(image_dict=ui.datasets.image_dict, percentage=percentage, cache=db)
    db.close()

    if show_logs:
        print('before filters!')
        for dataset_key in current_dataset['gt_images']:
            print(dataset_key, 'ARE THEY SAME?? ', len(current_dataset['gt_images'][dataset_key]), len(current_dataset['gt_images'][dataset_key]))
            match_counter = 0
            for i, j in zip(current_dataset['gt_images'][dataset_key], current_dataset['pred_images'][dataset_key]):
                # print(i['image_name'], j['image_name'], i['image_name'] == j['image_name'])
                if i['image_name'] == j['image_name']:
                    match_counter += 1
            print('matched images =', match_counter)

    filtered_classes = class_filtering(dataset=current_dataset, classes_names=classes_names)
    if show_logs:
        print('After filtered_classes!')
        for dataset_key in filtered_classes['gt_images']:
            print(dataset_key, 'ARE THEY SAME?? ', len(filtered_classes['gt_images'][dataset_key]),
                  len(filtered_classes['gt_images'][dataset_key]))
            match_counter = 0
            for i, j in zip(filtered_classes['gt_images'][dataset_key], filtered_classes['pred_images'][dataset_key]):
                if i['image_name'] == j['image_name']:
                    match_counter += 1
            print('matched images =', match_counter)

    filtered_confidences = confidence_filtering(dataset=filtered_classes, confidence_threshold=confidence_threshold)
    if show_logs:
        print('After filtered_confidences!')
        for dataset_key in filtered_confidences['gt_images']:
            print(dataset_key, 'ARE THEY SAME?? ', len(filtered_confidences['gt_images'][dataset_key]),
                  len(filtered_confidences['gt_images'][dataset_key]))
            match_counter = 0
            for i, j in zip(filtered_confidences['gt_images'][dataset_key], filtered_confidences['pred_images'][dataset_key]):
                if i['image_name'] == j['image_name']:
                    match_counter += 1
            print('matched images =', match_counter)

    plt_boxes = {}
    for prj_key, prj_value in filtered_confidences.items():
        plt_boxes[prj_key] = []
        bb_type = BBType.GROUND_TRUTH if prj_key == 'gt_images' else BBType.DETECTED
        for dataset_key, dataset_value in prj_value.items():
            for element in dataset_value:
                # print('element before bb = ', element)
                boxes = utils.plt2bb(batch_element=element, encoder=BoundingBox, bb_type=bb_type)
                plt_boxes[prj_key].extend(boxes)


def prepare_data(api: sly.Api, src_list, dst_list, encoder):
    gts = []
    pred = []
    dataset_names = {}
    for dataset_key in src_list:
        for gt_image, pr_image in zip(src_list[dataset_key], dst_list[dataset_key]):
            gt_boxes = utils.plt2bb(gt_image, encoder, bb_type=BBType.GROUND_TRUTH)
            pred_boxes = utils.plt2bb(pr_image, encoder, bb_type=BBType.DETECTED)

            if gt_image['dataset_id'] not in dataset_names:
                dataset_names[gt_image['dataset_id']] = api.dataset.get_info_by_id(gt_image['dataset_id']).name
            if pr_image['dataset_id'] not in dataset_names:
                dataset_names[pr_image['dataset_id']] = api.dataset.get_info_by_id(pr_image['dataset_id']).name

            gts.append([gt_image['image_id'], gt_image['image_name'], gt_image['full_storage_url'],
                        dataset_names[gt_image['dataset_id']], gt_boxes])
            pred.append([pr_image['image_id'], pr_image['image_name'], pr_image['full_storage_url'],
                         dataset_names[pr_image['dataset_id']], pred_boxes])
    return gts, pred, dataset_names


@g.my_app.callback("evaluate_button_click")
@sly.timeit
def evaluate_button_click(api: sly.Api, task_id, context, state, app_logger):
    global cm, gts, pred, dataset_names
    selected_classes = state['selectedClasses']
    percentage = state['samplePercent']
    iou_threshold = state['IoUThreshold']/100
    score_threshold = state['ScoreThreshold']/100

    if selected_classes:
        exec_download_v2(selected_classes, percentage=percentage, confidence_threshold=score_threshold)

    if plt_boxes['gt_images'] and plt_boxes['pred_images']:
        cm = calculate_confusion_matrix(gt=plt_boxes['gt_images'], det=plt_boxes['pred_images'],
                                        iou_threshold=iou_threshold, score_threshold=score_threshold,
                                        api=g.api, task_id=g.task_id)
        gts, pred, dataset_names = prepare_data(api=g.api,
                                                src_list=filtered_confidences['gt_images'],
                                                dst_list=filtered_confidences['pred_images'],
                                                encoder=RepoBoundingBox)
        method = metrics.MethodAveragePrecision.EVERY_POINT_INTERPOLATION

        overall_metrics.calculate_overall_metrics(api, task_id, gts, pred, g.pred_project_info.name, method,
                                                  iou_threshold, score_threshold)
        per_image_metrics.calculate_per_image_metrics(api, task_id, gts, pred, method,
                                                      iou_threshold, score_threshold)
        per_class_metrics.calculate_per_classes_metrics(api, task_id, gts, pred, g.pred_project_info.name, method,
                                                        iou_threshold, score_threshold)


@g.my_app.callback("view_class")
@sly.timeit
def view_class(api: sly.Api, task_id, context, state, app_logger):
    class_name = state["selectedClassName"]
    iou_threshold = state["IoUThreshold"]/100
    score_threshold = state['ScoreThreshold']/100
    per_class_metrics.selected_class_metrics(api, task_id, gts, pred, class_name, g.pred_project_info.name,
                                             iou_threshold, score_threshold)


total_image_num = dict()
current_dataset = dict()
filtered_classes = dict()
filtered_confidences = dict()
plt_boxes = dict()
cm = dict()
gts = {}
pred = {}
dataset_names = {}
