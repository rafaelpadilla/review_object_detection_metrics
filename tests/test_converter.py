import fnmatch
import json
import os

import pytest
import src.utils.converter as converter
import src.utils.general_utils as general_utils
import src.utils.validations as validations
from src.utils.enumerators import BBType, FileFormat


def test_converters_gts():
    # Defining paths with images and annotations
    images_dir = 'data/database/images'
    gts_dir = 'data/database/gts'
    assert os.path.isdir(images_dir)
    assert os.path.isdir(gts_dir)

    # COCO
    coco_dir = os.path.join(gts_dir, 'coco_format')
    coco_bbs = converter.coco2bb(coco_dir)
    coco_bbs.sort(key=lambda x: str(x), reverse=True)
    # CVAT
    cvat_dir = os.path.join(gts_dir, 'cvat_format')
    cvat_bbs = converter.cvat2bb(cvat_dir)
    cvat_bbs.sort(key=lambda x: str(x), reverse=True)
    # IMAGENET
    imagenet_dir = os.path.join(gts_dir, 'imagenet_format/Annotations')
    imagenet_bbs = converter.imagenet2bb(imagenet_dir)
    imagenet_bbs.sort(key=lambda x: str(x), reverse=True)
    # LABEL ME
    labelme_dir = os.path.join(gts_dir, 'labelme_format')
    labelme_bbs = converter.labelme2bb(labelme_dir)
    labelme_bbs.sort(key=lambda x: str(x), reverse=True)
    # OPEN IMAGE
    openimage_dir = os.path.join(gts_dir, 'openimages_format')
    openimage_bbs = converter.openimage2bb(openimage_dir, images_dir)
    openimage_bbs.sort(key=lambda x: str(x), reverse=True)
    # VOC PASCAL
    vocpascal_dir = os.path.join(gts_dir, 'pascalvoc_format')
    vocpascal_bbs = converter.vocpascal2bb(vocpascal_dir)
    vocpascal_bbs.sort(key=lambda x: str(x), reverse=True)
    # YOLO
    yolo_annotations_dir = os.path.join(gts_dir, 'yolo_format/obj_train_data')
    yolo_names_file = os.path.join(gts_dir, 'yolo_format/obj.names')
    yolo_bbs = converter.yolo2bb(yolo_annotations_dir,
                                 images_dir,
                                 yolo_names_file,
                                 bb_type=BBType.GROUND_TRUTH)
    yolo_bbs.sort(key=lambda x: str(x), reverse=True)

    assert len(coco_bbs) == len(cvat_bbs) == len(imagenet_bbs) == len(labelme_bbs) == len(
        openimage_bbs) == len(vocpascal_bbs) == len(yolo_bbs)

    for coco_bb, cvat_bb, imagenet_bb, labelme_bb, openimage_bb, vocpascal_bb, yolo_bb in zip(
            coco_bbs, cvat_bbs, imagenet_bbs, labelme_bbs, openimage_bbs, vocpascal_bbs, yolo_bbs):
        assert coco_bb == cvat_bb == imagenet_bb == labelme_bb == openimage_bb == vocpascal_bb == yolo_bb


def test_converters_dets():
    # Defining paths with images and annotations
    images_dir = 'data/database/images'
    gts_dir = 'data/database/dets'
    assert os.path.isdir(images_dir)
    assert os.path.isdir(gts_dir)

    # COCO
    coco_dir = os.path.join(gts_dir, 'coco_format')
    coco_bbs = converter.coco2bb(coco_dir)
    coco_bbs.sort(key=lambda x: str(x), reverse=True)
    # CVAT
    cvat_dir = os.path.join(gts_dir, 'cvat_format')
    cvat_bbs = converter.cvat2bb(cvat_dir)
    cvat_bbs.sort(key=lambda x: str(x), reverse=True)
    # IMAGENET
    imagenet_dir = os.path.join(gts_dir, 'imagenet_format/Annotations')
    imagenet_bbs = converter.imagenet2bb(imagenet_dir)
    imagenet_bbs.sort(key=lambda x: str(x), reverse=True)
    # LABEL ME
    labelme_dir = os.path.join(gts_dir, 'labelme_format')
    labelme_bbs = converter.labelme2bb(labelme_dir)
    labelme_bbs.sort(key=lambda x: str(x), reverse=True)
    # OPEN IMAGE
    openimage_dir = os.path.join(gts_dir, 'openimages_format')
    openimage_bbs = converter.openimage2bb(openimage_dir, images_dir)
    openimage_bbs.sort(key=lambda x: str(x), reverse=True)
    # VOC PASCAL
    vocpascal_dir = os.path.join(gts_dir, 'pascalvoc_format')
    vocpascal_bbs = converter.vocpascal2bb(vocpascal_dir)
    vocpascal_bbs.sort(key=lambda x: str(x), reverse=True)
    # YOLO
    yolo_annotations_dir = os.path.join(gts_dir, 'yolo_format/obj_train_data')
    yolo_names_file = os.path.join(gts_dir, 'yolo_format/obj.names')
    yolo_bbs = converter.yolo2bb(yolo_annotations_dir,
                                 images_dir,
                                 yolo_names_file,
                                 bb_type=BBType.GROUND_TRUTH)
    yolo_bbs.sort(key=lambda x: str(x), reverse=True)

    assert len(coco_bbs) == len(cvat_bbs) == len(imagenet_bbs) == len(labelme_bbs) == len(
        openimage_bbs) == len(vocpascal_bbs) == len(yolo_bbs)

    for coco_bb, cvat_bb, imagenet_bb, labelme_bb, openimage_bb, vocpascal_bb, yolo_bb in zip(
            coco_bbs, cvat_bbs, imagenet_bbs, labelme_bbs, openimage_bbs, vocpascal_bbs, yolo_bbs):
        assert coco_bb == cvat_bb == imagenet_bb == labelme_bb == openimage_bb == vocpascal_bb == yolo_bb


def test_toy_example_dets():
    dir_annots_dets = 'toyexample/dets/yolo_format'

    pascal_files = general_utils.get_files_recursively(dir_annots_dets)
    assert len(pascal_files) > 0
    for pascal_file in pascal_files:
        assert validations.is_yolo_format(pascal_file, bb_types=[BBType.DETECTED])


def test_toy_example_gts():
    dir_annots_dets = 'toyexample/gts'

    yolo_files = general_utils.get_files_recursively(dir_annots_dets)
    assert len(yolo_files) > 0
    for yolo_file in yolo_files:
        assert validations.is_pascal_format(yolo_file)
