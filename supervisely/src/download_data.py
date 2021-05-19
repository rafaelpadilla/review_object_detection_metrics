import os
import supervisely_lib as sly
import globals as g
import ui
import numpy as np
import random
import shelve
from src.utils.enumerators import MethodAveragePrecision
from supervisely.src.confusion_matrix.bounding_box_py import CoordinatesType, BBType, BBFormat, BoundingBox
from supervisely.src.ui.confusion_matrix import calculate_confusion_matrix
from supervisely.src.ui.metrics import calculate_metrics
import utils


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


def get_random_sample(image_dict, intersected_datasets, percentage, __sample=None):
    global total_image_num
    indexes = {}
    for key, value in image_dict['gt_images'].items():
        if key not in intersected_datasets:
            continue
        dataset_length = len(value)
        if __sample is None:
            total_image_num[key] = dataset_length
            sample_size = get_sample_size(dataset_length, percentage)
        else:
            sample_size = __sample
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
    # image = AnnotationInfo(image_id=876511, image_name='2008_003913.jpg',
    #                annotation={'description': '', 'tags': [], 'size': {'height': 500, 'width': 332}, 'objects': [
    #                    {'id': 26819549, 'classId': 32724, 'description': '', 'geometryType': 'rectangle',
    #                     'labelerLogin': 'DmitriyM', 'createdAt': '2021-05-14T09:19:18.069Z',
    #                     'updatedAt': '2021-05-14T09:19:18.069Z', 'tags': [], 'classTitle': 'boat',
    #                     'points': {'exterior': [[67, 93], [301, 423]], 'interior': []}}]},
    #                created_at='2021-05-14T09:19:18.069Z', updated_at='2021-05-14T09:19:18.069Z')
    new_image_list = []
    for image in image_list:
        # print('filter_classes: image =', image)
        # print('filter_classes: image.image_id =', image.image_id)

        bboxes = image['annotation']['objects']  # .annotation
        new_box_list = []
        for bbox in bboxes:
            if bbox['classTitle'] in classes_names:
                new_box_list.append(bbox)
        image['annotation']['objects'] = new_box_list
        new_image_list.append(image)
    return new_image_list


def filter_confidences(image_list, confidence_threshold):
    new_image_list =[]
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
            # image_id: {ImageInfo, Annotation_info}
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
    return sample


def class_filtering(dataset, classes_names):
    filtered_classes = {}
    for prj_key, prj_value in dataset.items():  # gt + pred
        filtered_classes[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            filtered_classes[prj_key][dataset_key] = filter_classes(dataset_value, classes_names)
    return filtered_classes


def confidence_filtering(dataset, confidence_threshold):
    filtered_confidence = {}
    for prj_key, prj_value in dataset.items():  # gt + pred
        filtered_confidence[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            if prj_key == 'gt_images':
                filtered_confidence[prj_key][dataset_key] = dataset_value
            if prj_key == 'pred_images':
                filtered_confidence[prj_key][dataset_key] = filter_confidences(dataset_value, confidence_threshold)
    return filtered_confidence


def exec_download_v2(classes_names, percentage, confidence_threshold, reload_always=False):
    global current_dataset, filtered_classes, filtered_confidences, plt_boxes
    db = shelve.open(filename='db', writeback=True)
    try:
        db['previous_percentage'] = db['current_percentage']
    except:
        db['previous_percentage'] = 0

    db['current_percentage'] = percentage
    if percentage != db['previous_percentage']:
        current_dataset = download_v2(image_dict=ui.datasets.image_dict, percentage=percentage, cache=db)
    db.close()
    filtered_classes = class_filtering(dataset=current_dataset, classes_names=classes_names)
    filtered_confidences = confidence_filtering(dataset=filtered_classes, confidence_threshold=confidence_threshold)

    plt_boxes = {}
    for prj_key, prj_value in filtered_confidences.items():
        plt_boxes[prj_key] = []
        bb_type = BBType.GROUND_TRUTH if prj_key == 'gt_images' else BBType.DETECTED
        for dataset_key, dataset_value in prj_value.items():
            for element in dataset_value:
                # print('element before bb = ', element)
                boxes = utils.plt2bb(batch_element=element, encoder=BoundingBox, bb_type=bb_type)
                plt_boxes[prj_key].extend(boxes)


@g.my_app.callback("evaluate_button_click")
@sly.timeit
def evaluate_button_click(api: sly.Api, task_id, context, state, app_logger):
    global cm
    selected_classes = state['selectedClasses']
    percentage = state['samplePercent']
    iou_threshold = state['IoUThreshold']/100
    score_threshold = state['ScoreThreshold']/100
    exec_download_v2(selected_classes, percentage=percentage, confidence_threshold=score_threshold, reload_always=True)

    cm = calculate_confusion_matrix(gt=plt_boxes['gt_images'], det=plt_boxes['pred_images'],
                                    iou_threshold=iou_threshold, score_threshold=score_threshold,
                                    api=g.api, task_id=g.task_id)
    calculate_metrics(api=g.api, task_id=g.task_id,
                      src_list=filtered_confidences['gt_images'], dst_list=filtered_confidences['pred_images'],
                      method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                      dst_project_name=g.pred_project_info.name,
                      iou_threshold=iou_threshold, score_threshold=score_threshold)


total_image_num = dict()
current_dataset = dict()
filtered_classes = dict()
filtered_confidences = dict()
plt_boxes = dict()
cm = dict()

# gt = dict()
# det = dict()

