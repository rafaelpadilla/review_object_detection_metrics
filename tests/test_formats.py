import src.utils.general_utils as utils
import src.utils.validations as validations
from src.utils.enumerators import BBType, CoordinatesType, FileFormat


def test_validation_formats():
    # Validate COCO format
    folder_annotations = 'data/database/dets/coco_format'
    assert validations.is_valid_coco_dir(folder_annotations)

    folder_annotations = 'data/database/gts/coco_format_v1'
    assert validations.is_valid_coco_dir(folder_annotations)

    folder_annotations = 'data/database/gts/coco_format_v2'
    assert validations.is_valid_coco_dir(folder_annotations)

    # Validate pascal format
    folder_annotations = 'data/database/gts/pascalvoc_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.is_pascal_format(file_path)

    # Validate absolute text formats for detections
    folders_annotations = ['data/database/dets/abs_xywh', 'data/database/dets/abs_xyx2y2']
    for folder_annotations in folders_annotations:
        bb_files = utils.get_files_recursively(folder_annotations)
        assert len(bb_files) > 0
        for file_path in bb_files:
            assert validations.is_specific_text_format(file_path, CoordinatesType.ABSOLUTE,
                                                       BBType.DETECTED)

    # Validate relative text formats for detections
    folder_annotations = 'data/database/dets/rel_xywh'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.is_specific_text_format(file_path, CoordinatesType.RELATIVE,
                                                   BBType.DETECTED)

    # Validate CVAT format
    folder_annotations = 'data/database/gts/cvat_format'
    assert validations.is_valid_cvat_dir(folder_annotations)

    # Validate OpenImageDataset (CSV)
    folder_annotations = 'data/database/gts/openimages_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.OPENIMAGE)

    # Validate ImageNet (XML)
    folder_annotations = 'data/database/gts/imagenet_format/Annotations'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.IMAGENET)

    # Validate LABEL ME format
    folder_annotations = 'data/database/gts/labelme_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.LABEL_ME)

    # Validate voc pascal files
    folder_annotations = 'data/database/dets/abs_xywh'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(
            file_path,
            FileFormat.ABSOLUTE_TEXT,
        )

    # Validate yolo files
    folder_annotations = 'data/database/gts/yolo_format/obj_train_data'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.YOLO)
