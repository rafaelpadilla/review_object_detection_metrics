import fnmatch
import json
import os

import src.utils.general_utils as utils
import src.utils.validations as validations
from src.bounding_box import BoundingBox
from src.utils.enumerators import FileFormat


def test_validation_formats():
    # Validate COCO format
    folder_annotations = 'data/database/detections/coco_format'
    text_files = utils.get_files_recursively(folder_annotations)
    for text_file in text_files:
        assert validations.verify_format(text_file, FileFormat.COCO)

    # Validate CVAT format
    folder_annotations = 'data/database/detections/cvat_format'
    text_files = utils.get_files_recursively(folder_annotations)
    for text_file in text_files:
        assert validations.verify_format(text_file, FileFormat.CVAT)

    # Validate OpenImageDataset (CSV)
    folder_annotations = 'data/database/detections/openimage_format'
    text_files = utils.get_files_recursively(folder_annotations)
    for text_file in text_files:
        assert validations.verify_format(text_file, FileFormat.OPENIMAGE)

    # Validate ImageNet (XML)
    folder_annotations = 'data/database/detections/imagenet_format'
    text_files = utils.get_files_recursively(folder_annotations)
    for text_file in text_files:
        assert validations.verify_format(text_file, FileFormat.IMAGENET)

    # Validate LABEL ME format
    folder_annotations = 'data/database/detections/labelme_format'
    text_files = utils.get_files_recursively(folder_annotations)
    for text_file in text_files:
        assert validations.verify_format(text_file, FileFormat.LABEL_ME)

    # Validate voc pascal files
    folder_annotations = 'data/database/detections/pascalvoc_format/Annotations'
    pascal_files = utils.get_files_recursively(folder_annotations)
    for pascal_file in pascal_files:
        assert validations.verify_format(pascal_file, FileFormat.PASCAL)

    # Faltando detecções no formato xywh
    # # Validate regular text files
    # folder_annotations = 'data/database/detections/xywh_format'
    # text_files = get_files_recursively(folder_annotations)
    # for text_file in text_files:
    #     assert validations.get_format(text_file) == FileFormat.TEXT

    # Validate yolo files
    folder_annotations = 'data/database/detections/yolo_format/obj_train_data'
    text_files = utils.get_files_recursively(folder_annotations)
    for text_file in text_files:
        assert validations.verify_format(text_file, FileFormat.YOLO)
