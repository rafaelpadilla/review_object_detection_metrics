import os
import supervisely_lib as sly
import globals as g
import ui
import numpy as np
import random
from supervisely.src.confusion_matrix.bounding_box_py import CoordinatesType, BBType, BBFormat, BoundingBox
from supervisely.src.confusion_matrix import confusion_matrix_py


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

        bboxes = image.annotation['objects']
        new_box_list = []
        for bbox in bboxes:
            if bbox['classTitle'] in classes_names:
                new_box_list.append(bbox)
        image.annotation['objects'] = new_box_list
        new_image_list.append(image)
    return new_image_list


def plt2bb(batch_element, encoder, type_coordinates=CoordinatesType.ABSOLUTE,
           bb_type=BBType.GROUND_TRUTH, _format=BBFormat.XYX2Y2):
        ret = []
        annotations = batch_element.annotation['objects']
        for ann in annotations:
            class_title = ann['classTitle']
            points = ann['points']['exterior']
            x1, y1 = points[0]
            x2, y2 = points[1]

            if x1 >= x2 or y1 >= y2:
                continue

            width = batch_element.annotation['size']['width']
            height = batch_element.annotation['size']['height']
            confidence = None if bb_type == BBType.GROUND_TRUTH else ann['tags'][0]['value']

            bb = encoder(image_name=batch_element.image_name, class_id=class_title,
                             coordinates=(x1, y1, x2, y2), type_coordinates=type_coordinates,
                             img_size=(width, height), confidence=confidence, bb_type=bb_type, format=_format)
            ret.append(bb)
        return ret


def filter_confidences(image_list, confidence_threshold):
    new_image_list =[]
    for image in image_list:
        # print('filter_confidences: image =', image)
        bboxes = image.annotation['objects']
        # print('bboxes =', bboxes)
        if len(bboxes) < 1:
            continue
        new_box_list = []
        for bbox in bboxes:
            if bbox['tags'][0]['value'] > confidence_threshold:
                new_box_list.append(bbox)
        image.annotation['objects'] = new_box_list
        new_image_list.append(image)
    return new_image_list


def match_objects():
    pass


def exec_download(classes_names, percentage, confidence_threshold, reload_always=False):
    global _sample_, total_image_num, plt_boxes, previous_percentage
    current_dataset = {}
    if reload_always:
        _sample_ = download(image_dict=ui.datasets.image_dict, percentage=percentage)
        current_dataset = _sample_.copy()
    else:
        if percentage!=0 and previous_percentage < percentage:
            previous_percentage = percentage
            _sample_ = download(image_dict=ui.datasets.image_dict, percentage=percentage)
            current_dataset = _sample_.copy()
        else:
            intersected_datasets = list(current_dataset.keys())
            current_sample_sizes = dict()
            for k, v in _sample_['gt_images'].items():
                current_sample_sizes[k] = get_sample_size(dataset_length=len(v), percentage=percentage)
            current_dataset = dict()
            random_indexes = get_random_sample(_sample_, intersected_datasets, percentage,
                                               __sample=current_sample_sizes)
            for prj_key, prj_value in _sample_.items():
                current_dataset[prj_key] = dict()
                for dataset_key, dataset_value in prj_value.items():
                    current_dataset[prj_key][dataset_key] = [_sample_[prj_key][dataset_key][index]
                                                             for index in random_indexes[dataset_key]]

    filtered_classes = {}
    for prj_key, prj_value in current_dataset.items():  # gt + pred
        filtered_classes[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            filtered_classes[prj_key][dataset_key] = filter_classes(dataset_value, classes_names)

    confidence_filtered_data = {}

    for prj_key, prj_value in filtered_classes.items():  # gt + pred
        confidence_filtered_data[prj_key] = {}
        for dataset_key, dataset_value in prj_value.items():
            if prj_key == 'gt_images':
                confidence_filtered_data[prj_key][dataset_key] = dataset_value
            if prj_key == 'pred_images':
                confidence_filtered_data[prj_key][dataset_key] = filter_confidences(dataset_value, confidence_threshold)

    plt_boxes = {}
    for prj_key, prj_value in filtered_classes.items():
        plt_boxes[prj_key] = []
        bb_type = BBType.GROUND_TRUTH if prj_key == 'gt_images' else BBType.DETECTED
        for dataset_key, dataset_value in prj_value.items():
            for element in dataset_value:
                boxes = plt2bb(batch_element=element, encoder=BoundingBox, bb_type=bb_type)
                plt_boxes[prj_key].extend(boxes)


class DataStorage:
    def __init__(self, project_dict):
        self.project_dict = project_dict
        self.data = {}

    def get_task_data(self):
        for prj_key, prj_data in self.project_dict.items():
            self.data[prj_key] = dict(project=prj_data)
            ws_to_team = {}
            datasets = g.api.dataset.get_list(prj_data.id)
            for dataset in datasets:
                dataset_name = dataset.name
                images = g.api.image.get_list(dataset.id)
                modified_images = []
                for image_info in images:
                    if prj_data.workspace_id not in ws_to_team:
                        ws_to_team[prj_data.workspace_id] = g.api.workspace.get_info_by_id(prj_data.workspace_id).team_id
                    meta = {
                        "team_id": ws_to_team[prj_data.workspace_id],
                        "workspace_id": prj_data.workspace_id,
                        "project_id": prj_data.id,
                        "project_name": prj_data.name,
                        "dataset_name": dataset.name,
                        "meta": image_info.meta
                    }
                    image_info = image_info._replace(meta=meta)
                    modified_images.append(image_info)
                self.data[prj_key][dataset_name] = {'dataset': dataset, 'images': modified_images}

    def download(self):

        pass

    def get_sample(self):
        pass

    @staticmethod
    def _get_all_images(api: sly.Api, project):
        ds_info = {}
        ds_images = {}
        ws_to_team = {}
        for dataset in api.dataset.get_list(project.id):
            ds_info[dataset.name] = dataset
            images = api.image.get_list(dataset.id)
            modified_images = []
            for image_info in images:
                if project.workspace_id not in ws_to_team:
                    ws_to_team[project.workspace_id] = api.workspace.get_info_by_id(project.workspace_id).team_id
                meta = {
                    "team_id": ws_to_team[project.workspace_id],
                    "workspace_id": project.workspace_id,
                    "project_id": project.id,
                    "project_name": project.name,
                    "dataset_name": dataset.name,
                    "meta": image_info.meta
                }
                image_info = image_info._replace(meta=meta)
                modified_images.append(image_info)
            ds_images[dataset.name] = modified_images
        return ds_info, ds_images


data_storage = DataStorage(image_dict=ui.datasets.image_dict)

_sample_ = dict()
total_image_num = dict()
plt_boxes = dict()
previous_percentage = 0

