import fnmatch
import json
import os

import pytest
import src.utils.general_utils as utils
import src.utils.validations as validations
from src.bounding_box import BoundingBox
from src.utils.enumerators import FileFormat


def test_validation_formats():
    # Validate COCO format
    folder_annotations = 'data/database/detections/coco_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.COCO)

    # Validate CVAT format
    folder_annotations = 'data/database/detections/cvat_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.CVAT)

    # Validate OpenImageDataset (CSV)
    folder_annotations = 'data/database/detections/openimage_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.OPENIMAGE)

    # Validate ImageNet (XML)
    folder_annotations = 'data/database/detections/imagenet_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.IMAGENET)

    # Validate LABEL ME format
    folder_annotations = 'data/database/detections/labelme_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.LABEL_ME)

    # Validate voc pascal files
    folder_annotations = 'data/database/detections/pascalvoc_format/Annotations'
    pascal_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for pascal_file in pascal_files:
        assert validations.verify_format(pascal_file, FileFormat.PASCAL)

    # TODO: Faltando detecções no formato xywh
    # # Validate regular text files
    # folder_annotations = 'data/database/detections/xywh_format'
    # bb_files = get_files_recursively(folder_annotations)
    # for file_path in bb_files:
    #     assert validations.get_format(file_path) == FileFormat.TEXT

    # Validate yolo files
    folder_annotations = 'data/database/detections/yolo_format/obj_train_data'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.YOLO)
