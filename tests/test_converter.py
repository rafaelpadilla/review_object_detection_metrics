import os

import src.utils.converter as converter
import src.utils.general_utils as general_utils
import src.utils.validations as validations
from src.utils.enumerators import BBFormat, BBType, CoordinatesType


def test_converters_gts():
    # Defining paths with images and annotations
    images_dir = 'data/database/images'
    gts_dir = 'data/database/gts'
    assert os.path.isdir(images_dir)
    assert os.path.isdir(gts_dir)

    # COCO
    coco_dir = os.path.join(gts_dir, 'coco_format_v1')
    coco_bbs_v1 = converter.coco2bb(coco_dir)
    coco_bbs_v1.sort(key=lambda x: str(x), reverse=True)
    # COCO
    coco_dir = os.path.join(gts_dir, 'coco_format_v2')
    coco_bbs_v2 = converter.coco2bb(coco_dir)
    coco_bbs_v2.sort(key=lambda x: str(x), reverse=True)
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

    assert len(coco_bbs_v1) == len(coco_bbs_v2) == len(cvat_bbs) == len(imagenet_bbs) == len(
        labelme_bbs) == len(openimage_bbs) == len(vocpascal_bbs) == len(yolo_bbs)

    for coco_bb_v1, coco_bb_v2, cvat_bb, imagenet_bb, labelme_bb, openimage_bb, vocpascal_bb, yolo_bb in zip(
            coco_bbs_v1, coco_bbs_v2, cvat_bbs, imagenet_bbs, labelme_bbs, openimage_bbs,
            vocpascal_bbs, yolo_bbs):
        assert coco_bb_v1 == coco_bb_v2 == cvat_bb == imagenet_bb == labelme_bb == openimage_bb == vocpascal_bb == yolo_bb


def test_converters_dets():
    # Defining paths with images and annotations
    images_dir = 'data/database/images'
    gts_dir = 'data/database/dets'
    assert os.path.isdir(images_dir)
    assert os.path.isdir(gts_dir)

    # COCO
    coco_dir = os.path.join(gts_dir, 'coco_format_v1')
    coco_bbs_v1 = converter.coco2bb(coco_dir)
    coco_bbs_v1.sort(key=lambda x: str(x), reverse=True)
    # COCO
    coco_dir = os.path.join(gts_dir, 'coco_format_v2')
    coco_bbs_v2 = converter.coco2bb(coco_dir)
    coco_bbs_v2.sort(key=lambda x: str(x), reverse=True)
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

    assert len(coco_bbs_v1) == len(coco_bbs_v2) == len(cvat_bbs) == len(imagenet_bbs) == len(
        labelme_bbs) == len(openimage_bbs) == len(vocpascal_bbs) == len(yolo_bbs)

    for coco_bb_v1, coco_bb_v2, cvat_bb, imagenet_bb, labelme_bb, openimage_bb, vocpascal_bb, yolo_bb in zip(
            coco_bbs_v1, coco_bbs_v2, cvat_bbs, imagenet_bbs, labelme_bbs, openimage_bbs,
            vocpascal_bbs, yolo_bbs):
        assert coco_bb_v1 == cvat_bb == imagenet_bb == labelme_bb == openimage_bb == vocpascal_bb == yolo_bb


def test_toy_example_dets():
    dir_annots_dets = 'toyexample/dets_classid_rel_xcycwh'

    yolo_files = general_utils.get_files_recursively(dir_annots_dets)
    assert len(yolo_files) > 0
    for pascal_file in yolo_files:
        assert validations.is_yolo_format(pascal_file, bb_types=[BBType.DETECTED])


def test_toy_example_dets_compatibility():
    # Checks if all different formats in the toyexample represent the same coordinates
    dir_img_dir = 'toyexample/images'
    filepath_classes_det = 'toyexample/voc.names'

    dir_dets_classid_abs_xywh = 'toyexample/dets_classid_abs_xywh'
    dets_classid_abs_xywh = converter.text2bb(dir_dets_classid_abs_xywh,
                                              bb_type=BBType.DETECTED,
                                              bb_format=BBFormat.XYWH,
                                              type_coordinates=CoordinatesType.ABSOLUTE)
    dets_classid_abs_xywh = general_utils.replace_id_with_classes(dets_classid_abs_xywh,
                                                                  filepath_classes_det)
    dets_classid_abs_xywh.sort(key=lambda x: str(x), reverse=True)

    dir_dets_classid_abs_xyx2y2 = 'toyexample/dets_classid_abs_xyx2y2'
    dets_classid_abs_xyx2y2 = converter.text2bb(dir_dets_classid_abs_xyx2y2,
                                                bb_type=BBType.DETECTED,
                                                bb_format=BBFormat.XYX2Y2,
                                                type_coordinates=CoordinatesType.ABSOLUTE)
    dets_classid_abs_xyx2y2 = general_utils.replace_id_with_classes(dets_classid_abs_xyx2y2,
                                                                    filepath_classes_det)
    dets_classid_abs_xyx2y2.sort(key=lambda x: str(x), reverse=True)

    dir_dets_classid_rel_xcycwh = 'toyexample/dets_classid_rel_xcycwh'
    dets_classid_rel_xcycwh = converter.text2bb(dir_dets_classid_rel_xcycwh,
                                                bb_type=BBType.DETECTED,
                                                bb_format=BBFormat.YOLO,
                                                type_coordinates=CoordinatesType.RELATIVE,
                                                img_dir=dir_img_dir)
    dets_classid_rel_xcycwh = general_utils.replace_id_with_classes(dets_classid_rel_xcycwh,
                                                                    filepath_classes_det)
    dets_classid_rel_xcycwh.sort(key=lambda x: str(x), reverse=True)

    dir_dets_classname_abs_xywh = 'toyexample/dets_classname_abs_xywh'
    dets_classname_abs_xywh = converter.text2bb(dir_dets_classname_abs_xywh,
                                                bb_type=BBType.DETECTED,
                                                bb_format=BBFormat.XYWH,
                                                type_coordinates=CoordinatesType.ABSOLUTE)
    dets_classname_abs_xywh.sort(key=lambda x: str(x), reverse=True)

    dir_dets_classname_abs_xyx2y2 = 'toyexample/dets_classname_abs_xyx2y2'
    dets_classname_abs_xyx2y2 = converter.text2bb(dir_dets_classname_abs_xyx2y2,
                                                  bb_type=BBType.DETECTED,
                                                  bb_format=BBFormat.XYX2Y2,
                                                  type_coordinates=CoordinatesType.ABSOLUTE)
    dets_classname_abs_xyx2y2.sort(key=lambda x: str(x), reverse=True)

    dir_dets_classname_rel_xcycwh = 'toyexample/dets_classname_rel_xcycwh'
    dets_classname_rel_xcycwh = converter.text2bb(dir_dets_classname_rel_xcycwh,
                                                  bb_type=BBType.DETECTED,
                                                  bb_format=BBFormat.YOLO,
                                                  type_coordinates=CoordinatesType.RELATIVE,
                                                  img_dir=dir_img_dir)
    dets_classname_rel_xcycwh.sort(key=lambda x: str(x), reverse=True)

    dir_dets_coco_format = 'toyexample/dets_coco_format'
    dets_coco_format = converter.coco2bb(dir_dets_coco_format, bb_type=BBType.DETECTED)
    dets_coco_format.sort(key=lambda x: str(x), reverse=True)

    for a, b, c, d, e, f, g in zip(dets_classid_abs_xywh, dets_classid_abs_xyx2y2,
                                   dets_classid_rel_xcycwh, dets_classname_abs_xywh,
                                   dets_classname_abs_xyx2y2, dets_classname_rel_xcycwh,
                                   dets_coco_format):
        assert a == b == c == d == e == f == g


def test_toy_example_gts():
    ############################################################################
    # Verify if all files in the toy example follow their expected format
    ############################################################################

    # PASCAL VOC
    dir_annots_gts_pascal = 'toyexample/gts_vocpascal_format'
    files = general_utils.get_files_recursively(dir_annots_gts_pascal)
    assert len(files) > 0
    for f in files:
        assert validations.is_pascal_format(
            f), 'File {f} does not follow the expected format (PASCAL VOC)'

    # COCO
    dir_annots_gts_coco = 'toyexample/gts_coco_format'
    files = general_utils.get_files_recursively(dir_annots_gts_coco)
    assert len(files) > 0
    for f in files:
        assert validations.is_coco_format(f), 'File {f} does not follow the expected format (COCO)'

    ############################################################################
    # Compare if all bounding boxes are the same
    ############################################################################
    pascal_bbs = converter.vocpascal2bb(dir_annots_gts_pascal)
    coco_bbs = converter.coco2bb(dir_annots_gts_coco)

    coco_bbs.sort(key=lambda x: str(x), reverse=True)
    pascal_bbs.sort(key=lambda x: str(x), reverse=True)

    assert len(coco_bbs) == len(pascal_bbs)

    for coco_bb, pascal_bb in zip(coco_bbs, pascal_bbs):
        assert coco_bb == pascal_bb
