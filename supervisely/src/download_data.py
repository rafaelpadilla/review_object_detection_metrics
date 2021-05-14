import os
import supervisely_lib as sly
import globals as g
import ui
import numpy as np
import random

slice_ = dict()
total_image_num = dict()


def get_intersected_datasets(image_dict):
    return list(set(list(image_dict['gt_images'])) & set(list(image_dict['pred_images'])))


def get_slice(image_dict, intersected_datasets, percentage):
    indexes = {}
    for key, value in image_dict['gt_images'].items():
        if key not in intersected_datasets:
            continue
        dataset_length = len(value)
        total_image_num[key] = dataset_length
        sample_size = int(np.ceil(dataset_length / 100 * percentage))
        indexes[key] = random.sample(range(dataset_length), sample_size)
    return indexes


def download(image_dict, percentage, batch_size=10):
    global slice_, total_image_num

    intersected_datasets = get_intersected_datasets(image_dict)
    indexes = get_slice(image_dict, intersected_datasets, percentage)

    for project_key, project_info in image_dict.items():
        slice_[project_key] = []
        for dataset_key, dataset_info in project_info.items():
            if dataset_key not in intersected_datasets:
                continue
            dataset_id = dataset_info[0].dataset_id
            slice_to_download = [dataset_info[index] for index in indexes[dataset_key]]
            for ix, batch in enumerate(sly.batched(slice_to_download, batch_size)):
                image_ids = [image_info.id for image_info in batch]
                annotations = g.api.annotation.download_batch(dataset_id, image_ids)
                slice_[project_key].extend(annotations)


def convert_sly_to_bb():
    pass


def filter_classes(annotation, classes_names):
    pass


def filter_confidences():
    pass


def match_objects():
    pass


class ConfusionMatrix:
    def __init__(self):
        pass

