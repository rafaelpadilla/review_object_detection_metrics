import numpy as np
import random
import shelve
import supervisely_lib as sly
import globals as g
import ui
from src.bounding_box import BoundingBox as RepoBoundingBox, BBType
from supervisely.src.ui import metrics, overall_metrics, per_image_metrics, per_class_metrics
# from supervisely_lib.app.widgets.confusion_matrix import ConfusionMatrix, plt2bb
from utils import plt2bb
from widgets.confusion_matrix import ConfusionMatrix


def get_filtered_image_names(filtered_dict, key):
    gt_image_names = [image['image_name'] for image in filtered_dict['gt_images'][key]]
    pred_image_names = [image['image_name'] for image in filtered_dict['pred_images'][key]]
    only_in_gt = [item for item in gt_image_names if item not in pred_image_names]
    only_in_pred = [item for item in pred_image_names if item not in gt_image_names]
    only_in = {'gt_images': only_in_gt, 'pred_images': only_in_pred}
    return only_in


def check_for_class_existence(image, classes_names, check_for_obj_existence=False):
    bboxes = image['annotation'].labels
    new_box_list = []
    for bbox in bboxes:
        if bbox.obj_class.name in classes_names:
            new_box_list.append(bbox)
    image['annotation'] = image['annotation'].clone(labels=new_box_list)
    if not check_for_obj_existence:
        return image
    else:
        existence_flag = True if new_box_list else False
        return image, existence_flag


def check_for_conf_threshold(image, conf_threshold, check_for_obj_existence=False):
    bboxes = image['annotation'].labels
    new_box_list = []
    if len(bboxes) < 1:
        if not check_for_obj_existence:
            return image
        else:
            existence_flag = True if new_box_list else False
            return image, existence_flag
    for bbox in bboxes:
        if bbox.tags.get('confidence').value > conf_threshold:
            new_box_list.append(bbox)
    image['annotation'] = image['annotation'].clone(labels=new_box_list)
    if not check_for_obj_existence:
        return image
    else:
        existence_flag = True if new_box_list else False
        return image, existence_flag


def get_additional_images(additional_image_list, task_tuple):
    new_image_list = []
    task_name = task_tuple[0]
    checker = check_for_class_existence if task_name == 'class_filtering' else check_for_conf_threshold
    task_param = task_tuple[1]
    for image in additional_image_list:
        image = checker(image, task_param)
        new_image_list.append(image)
    return new_image_list


def class_filtering(dataset, classes_names):
    def filter_classes(image_list):
        new_image_list = []
        for image in image_list:
            image, existence_flag = check_for_class_existence(image, classes_names, True)
            if existence_flag:
                new_image_list.append(image)
        return new_image_list

    filtered_class = {}
    for prj_key, prj_value in dataset.items():  # gt + pred
        filtered_class[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            filtered_class[prj_key][dataset_key] = filter_classes(dataset_value)

    for dataset_key in dataset[list(dataset.keys())[0]]:
        additional_image_names = {}
        image_names_to_sort = {}
        indexes_ = {}
        # only_in = {'gt_images': only_in_gt, 'pred_images': only_in_pred}
        only_in = get_filtered_image_names(filtered_class, dataset_key)
        for key in ['gt_images', 'pred_images']:
            reversed_key = 'gt_images' if key=='pred_images' else 'pred_images'
            additional_image_names[key] = [
                image_info for image_info in dataset[key][dataset_key]
                if image_info['image_name'] in only_in[reversed_key]
            ]
            task_tuple = ('class_filtering', classes_names)
            additional_image_names[key] = get_additional_images(additional_image_names[key], task_tuple)
            filtered_class[key][dataset_key].extend(additional_image_names[key])
            image_names_to_sort[key] = [
                image_info['image_name'] for image_info in filtered_class['gt_images'][dataset_key]
            ]
            indexes_[key] = np.argsort(image_names_to_sort[key])
            filtered_class[key][dataset_key] = [
                filtered_class[key][dataset_key][i] for i in indexes_[key]
            ]
    return filtered_class


def confidence_filtering(dataset, confidence_threshold):
    def filter_confidences(image_list):
        new_image_list = []
        for image in image_list:
            image, existence_flag = check_for_conf_threshold(image, confidence_threshold, True)
            if existence_flag:
                new_image_list.append(image)
        return new_image_list

    filtered_confidence = {}
    for prj_key, prj_value in dataset.items():  # gt + pred
        filtered_confidence[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            if prj_key == 'gt_images':
                filtered_confidence[prj_key][dataset_key] = dataset_value
            if prj_key == 'pred_images':
                filtered_confidence[prj_key][dataset_key] = filter_confidences(dataset_value)

    for dataset_key in dataset[list(dataset.keys())[0]]:
        additional_image_names = {}
        image_names_to_sort = {}
        indexes_ = {}
        only_in = get_filtered_image_names(filtered_confidence, dataset_key)
        for key in ['gt_images', 'pred_images']:
            reversed_key = 'gt_images' if key=='pred_images' else 'pred_images'
            additional_image_names[key] = [
                image_info for image_info in dataset[key][dataset_key]
                if image_info['image_name'] in only_in[reversed_key]
            ]
            task_tuple = ('conf_filtering', confidence_threshold)
            if key=='pred_images':
                additional_image_names[key] = get_additional_images(additional_image_names[key], task_tuple)

            filtered_confidence[key][dataset_key].extend(additional_image_names[key])
            image_names_to_sort[key] = [
                image_info['image_name'] for image_info in filtered_confidence['gt_images'][dataset_key]
            ]
            indexes_[key] = np.argsort(image_names_to_sort[key])
            filtered_confidence[key][dataset_key] = [
                filtered_confidence[key][dataset_key][i] for i in indexes_[key]
            ]
    return filtered_confidence


def download(image_dict, percentage, cache, batch_size=10, show_info=False):
    def get_intersected_datasets(img_dict):
        gt_images = set(list(img_dict['gt_images']))
        pred_images = set(list(img_dict['pred_images']))
        intersected_datasets = list(gt_images & pred_images)
        return intersected_datasets

    def get_random_sample(image_dict, intersected_datasets, percentage):

        def get_sample_size(dataset_length, percentage):
            sample_size = int(np.ceil(dataset_length / 100 * percentage))
            return sample_size

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
                ann_infos = g.api.annotation.download_batch(dataset_id, image_ids)
                annotations = [sly.Annotation.from_json(ann_info.annotation, g.aggregated_meta) for ann_info in ann_infos]

                for batch_, annotation in zip(batch, annotations):
                    batch_image_id = batch_.id
                    dict_ = dict(image_id=batch_.id,
                                 image_name=batch_.name,
                                 labels_count=batch_.labels_count,
                                 dataset_id=batch_.dataset_id,
                                 annotation=annotation,
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
    filtered_classes = class_filtering(dataset=current_dataset, classes_names=classes_names)
    filtered_confidences = confidence_filtering(dataset=filtered_classes, confidence_threshold=confidence_threshold)
    return filtered_confidences


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


@g.my_app.callback("view_class")
@sly.timeit
def view_class(api: sly.Api, task_id, context, state, app_logger):
    print('state =', state)
    class_name = state["selectedClassName"]
    iou_threshold = state["IoUThreshold"] / 100
    score_threshold = state['ScoreThreshold'] / 100
    per_class_metrics.selected_class_metrics(api, task_id, gts, pred, class_name, g.pred_project_info.name,
                                             iou_threshold, score_threshold)
    fields = [
        {"field": "state.perClassActiveStep", "payload": 2}
    ]
    api.app.set_fields(task_id, fields)


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
