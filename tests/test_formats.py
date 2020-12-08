import src.utils.general_utils as utils
import src.utils.validations as validations
from src.utils.enumerators import FileFormat


def test_validation_formats():
    # Validate COCO format
    folder_annotations = 'data/database/dets/coco_format_v1'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.COCO)

    folder_annotations = 'data/database/dets/coco_format_v2'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.COCO)

    # Validate CVAT format
    folder_annotations = 'data/database/dets/cvat_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.CVAT)

    # Validate OpenImageDataset (CSV)
    folder_annotations = 'data/database/dets/openimages_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.OPENIMAGE)

    # Validate ImageNet (XML)
    folder_annotations = 'data/database/dets/imagenet_format/Annotations'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.IMAGENET)

    # Validate LABEL ME format
    folder_annotations = 'data/database/dets/labelme_format'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.LABEL_ME)

    # Validate voc pascal files
    folder_annotations = 'data/database/dets/pascalvoc_format/Annotations'
    pascal_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for pascal_file in pascal_files:
        assert validations.verify_format(pascal_file, FileFormat.PASCAL)

    # Validate yolo files
    folder_annotations = 'data/database/dets/yolo_format/obj_train_data'
    bb_files = utils.get_files_recursively(folder_annotations)
    assert len(bb_files) > 0
    for file_path in bb_files:
        assert validations.verify_format(file_path, FileFormat.YOLO)
