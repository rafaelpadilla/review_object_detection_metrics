import supervisely_lib as sly
import globals as g
import ui
import numpy as np
import random
import shelve
from src.bounding_box import BoundingBox as RepoBoundingBox, BBType
from supervisely.src.ui import metrics, overall_metrics, per_image_metrics, per_class_metrics
# from supervisely_lib.app.widgets.confusion_matrix import ConfusionMatrix, plt2bb
from widgets.confusion_matrix import ConfusionMatrix
from utils import plt2bb


def get_intersected_datasets(img_dict):
    gt_images = set(list(img_dict['gt_images']))
    pred_images = set(list(img_dict['pred_images']))
    intersected_datasets = list(gt_images & pred_images)
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
        # Chooses k unique random elements from a population sequence or set
        indexes[key] = random.sample(range(dataset_length), sample_size)
    return indexes


def check_for_existence(dataset):
    tmp_dict = dict()
    for prj_key, prj_value in dataset.items():
        tmp_dict[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            existing_names = []
            tmp_dict[prj_key][dataset_key] = list()
            for value in dataset_value:
                print('value = ', value)
                if value['image_name'] not in existing_names:
                    existing_names.append(value['image_name'])
                    tmp_dict[prj_key][dataset_key].append(value)
    return tmp_dict


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


def fn(additional_gt_images, classes_names):
    new_image_list = []
    for image in additional_gt_images:
        bboxes = image['annotation']['objects']  # .annotation
        new_box_list = []
        for bbox in bboxes:
            if bbox['classTitle'] in classes_names:
                new_box_list.append(bbox)
        image['annotation']['objects'] = new_box_list
        new_image_list.append(image)
    return new_image_list


def class_filtering(dataset, classes_names):
    filtered_class = {}
    for prj_key, prj_value in dataset.items():  # gt + pred
        filtered_class[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            filtered_class[prj_key][dataset_key] = filter_classes(dataset_value, classes_names)

    for dataset_key in dataset[list(dataset.keys())[0]]:
        gt_image_names = [image['image_name'] for image in filtered_class['gt_images'][dataset_key]]
        pred_image_names = [image['image_name'] for image in filtered_class['pred_images'][dataset_key]]
        list1 = gt_image_names
        list2 = pred_image_names
        only_in_first = [item for item in list1 if item not in list2]
        only_in_second = [item for item in list2 if item not in list1]

        additional_gt_images = [
            image_info for image_info in dataset['gt_images'][dataset_key]
            if image_info['image_name'] in only_in_second
        ]
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

        additional_pred_images = [
            image_info for image_info in dataset['pred_images'][dataset_key]
            if image_info['image_name'] in only_in_first
        ]
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

        gt_image_names_to_sort = [
            image_info['image_name'] for image_info in filtered_class['gt_images'][dataset_key]
        ]
        pred_image_names_to_sort = [
            image_info['image_name'] for image_info in filtered_class['pred_images'][dataset_key]
        ]

        indexes_gt = np.argsort(gt_image_names_to_sort)
        indexes_pred = np.argsort(pred_image_names_to_sort)
        filtered_class['gt_images'][dataset_key] = [
            filtered_class['gt_images'][dataset_key][i] for i in indexes_gt
        ]
        filtered_class['pred_images'][dataset_key] = [
            filtered_class['pred_images'][dataset_key][i] for i in indexes_pred
        ]
    return filtered_class


def filter_confidences(image_list, confidence_threshold):
    new_image_list = []
    for image in image_list:
        bboxes = image['annotation']['objects']
        if len(bboxes) < 1:
            continue
        new_box_list = []
        for bbox in bboxes:
            if bbox['tags'][0]['value'] > confidence_threshold:
                new_box_list.append(bbox)
        image['annotation']['objects'] = new_box_list
        new_image_list.append(image)
    return new_image_list


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
        list1 = gt_image_names
        list2 = pred_image_names
        only_in_first = [item for item in list1 if item not in list2]
        only_in_second = [item for item in list2 if item not in list1]
        additional_gt_images = [image_info for image_info in dataset['gt_images'][dataset_key] if
                                image_info['image_name'] in only_in_second]
        additional_pred_images = [image_info for image_info in dataset['pred_images'][dataset_key] if
                                  image_info['image_name'] in only_in_first]
        new_image_list = []
        for image in additional_pred_images:
            bboxes = image['annotation']['objects']
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


def download(image_dict, percentage, cache, batch_size=10, show_info=False):
    intersected_datasets = get_intersected_datasets(image_dict)
    sample = {}
    indexes = get_random_sample(image_dict, intersected_datasets, percentage)
    for project_key, project_info in image_dict.items():  # project_key in []
        sample[project_key] = dict()
        for dataset_key, dataset_info in project_info.items():
            if dataset_key not in intersected_datasets:
                continue
            sample[project_key][dataset_key] = list()
            dataset_id = dataset_info[0].dataset_id

            sample[project_key][dataset_key] = [
                cache[str(dataset_info[index].id)]
                for index in indexes[dataset_key]
                if str(dataset_info[index].id) in cache
            ]
            to_download_indexes = [index for index in indexes[dataset_key] if str(dataset_info[index].id) not in cache]
            slice_to_download = [dataset_info[index] for index in to_download_indexes]
            for ix, batch in enumerate(sly.batched(slice_to_download, batch_size)):
                image_ids = [image_info.id for image_info in batch]
                annotations = g.api.annotation.download_batch(dataset_id, image_ids)
                for batch_, annotation in zip(batch, annotations):
                    batch_image_id = batch_.id
                    annotation_image_id = annotation.image_id
                    assert batch_image_id == annotation_image_id, 'different images'
                    dict_ = dict(image_id=batch_.id, image_name=batch_.name, labels_count=batch_.labels_count,
                                 dataset_id=batch_.dataset_id, annotation=annotation.annotation,
                                 full_storage_url=batch_.full_storage_url)
                    cache[str(batch_image_id)] = dict_
                    sample[project_key][dataset_key].append(dict_)
            image_names = [d['image_name'] for d in sample[project_key][dataset_key]]
            indexes_to_sort = np.argsort(image_names)
            sample[project_key][dataset_key] = [sample[project_key][dataset_key][index] for index in indexes_to_sort]
    return sample


def download_and_prepare_data(classes_names, percentage, confidence_threshold):
    global current_dataset, filtered_classes, filtered_confidences
    db = shelve.open(filename='db', writeback=True)
    current_dataset = download(image_dict=ui.datasets.image_dict, percentage=percentage, cache=db)
    db.close()
    # current_dataset = check_for_existence(current_dataset)
    filtered_classes = class_filtering(dataset=current_dataset, classes_names=classes_names)
    filtered_confidences = confidence_filtering(dataset=filtered_classes, confidence_threshold=confidence_threshold)


def get_prepared_data(api: sly.Api, src_list, dst_list, encoder):
    gts = []
    pred = []
    dataset_names = {}
    for dataset_key in src_list:
        for gt_image, pr_image in zip(src_list[dataset_key], dst_list[dataset_key]):
            gt_boxes = plt2bb(gt_image, encoder, bb_type=BBType.GROUND_TRUTH)
            pred_boxes = plt2bb(pr_image, encoder, bb_type=BBType.DETECTED)

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
    global cm, gts, pred, dataset_names, previous_percentage
    selected_classes = state['selectedClasses']
    percentage = state['samplePercent']
    iou_threshold = state['IoUThreshold'] / 100
    score_threshold = state['ScoreThreshold'] / 100

    if selected_classes:
        if percentage != previous_percentage:
            download_and_prepare_data(selected_classes, percentage=percentage, confidence_threshold=score_threshold)
            previous_percentage = percentage

        if filtered_confidences['gt_images'] and filtered_confidences['pred_images']:
            confusion_matrix.set_data(gt=filtered_confidences['gt_images'], det=filtered_confidences['pred_images'])
            confusion_matrix.reset_thresholds(iou_threshold=iou_threshold, score_threshold=score_threshold)
            confusion_matrix.update()
            cm = confusion_matrix.cm_dict

            gts, pred, dataset_names = get_prepared_data(api=g.api,
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
    print('state =', state)
    class_name = state["selectedClassName"]
    iou_threshold = state["IoUThreshold"] / 100
    score_threshold = state['ScoreThreshold'] / 100
    per_class_metrics.selected_class_metrics(api, task_id, gts, pred, class_name, g.pred_project_info.name,
                                             iou_threshold, score_threshold)


total_image_num = dict()
current_dataset = dict()
filtered_classes = dict()
filtered_confidences = dict()
cm = dict()
gts = {}
pred = {}
dataset_names = {}
previous_percentage = 0
confusion_matrix = ConfusionMatrix(api=g.api, task_id=g.task_id, v_model='data.slyConfusionMatrix')
