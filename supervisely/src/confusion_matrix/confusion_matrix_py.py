# @TODO: create class for confusion matrix
import numpy as np
from .bounding_box_py import BoundingBox


def collect_data(gt_boxes, det_boxes, score_threshold):
    _gt_boxes = gt_boxes
    _det_boxes = list()
    for box in det_boxes:
        if box.get_confidence() >= score_threshold:
            _det_boxes.append(box)

    gt_images_only = []
    classes_bbs = {}
    if len(gt_boxes) == 0:
        pass  # return

    for bb in _gt_boxes:
        image_name = bb.get_image_name()
        gt_images_only.append(image_name)
        classes_bbs.setdefault(image_name, {'gt': [], 'det': []})
        classes_bbs[image_name]['gt'].append(bb)

    for bb in _det_boxes:
        image_name = bb.get_image_name()
        classes_bbs.setdefault(image_name, {'gt': [], 'det': []})
        classes_bbs[image_name]['det'].append(bb)

    classes = [item.get_class_id() for item in _gt_boxes] + [item.get_class_id() for item in _det_boxes]
    classes = list(set(classes))
    classes.sort()
    classes.append('None')

    # row = annotations, col = detections
    conf_matrix = dict()
    for class_row in classes:
        conf_matrix[class_row] = {}
        for class_col in classes:
            conf_matrix[class_row][class_col] = list()

    return _gt_boxes, _det_boxes, conf_matrix, classes_bbs


def confusion_matrix(gt_boxes, det_boxes, iou_threshold=0.5, score_threshold=0.01):
    # print('confusion_matrix: gt_boxes =', gt_boxes[0])
    _gt_boxes, _det_boxes, conf_matrix, classes_bbs = collect_data(gt_boxes, det_boxes, score_threshold)
    for image, data in classes_bbs.items():
        anns = data['gt']
        dets = data['det']
        dets = [a for a in sorted(dets, key=lambda bb: bb.get_confidence(), reverse=True)]
        if len(anns) != 0:
            if len(dets) != 0:  # annotations - yes, detections - yes:
                iou_matrix = np.zeros((len(dets), len(anns)))
                for det_id, det in enumerate(dets):
                    for ann_id, ann in enumerate(anns):
                        iou_matrix[det_id, ann_id] = BoundingBox.iou(det, ann)
                detected_gt_per_image = np.zeros(len(anns))
                for det_idx in range(iou_matrix.shape[0]):
                    ann_idx = np.argmax(iou_matrix[det_idx])
                    iou_value = iou_matrix[det_idx, ann_idx]
                    if iou_value >= iou_threshold:
                        if detected_gt_per_image[ann_idx] == 0:
                            detected_gt_per_image[ann_idx] = 1
                            ann_box = anns[ann_idx]
                            det_box = dets[det_idx]
                        else:
                            if np.sum(detected_gt_per_image) < detected_gt_per_image.shape[0]:
                                ann_box = anns[ann_idx]
                                det_box = 'None'
                            else:
                                ann_box = 'None'
                                det_box = dets[ann_idx]
                    else:
                        ann_box = 'None'  # anns[ann_idx]
                        det_box = dets[det_idx]
                    ann_cls = ann_box.get_class_id() if not isinstance(ann_box, str) else 'None'
                    det_cls = det_box.get_class_id() if not isinstance(det_box, str) else 'None'
                    conf_matrix[det_cls][ann_cls].append(image)
                if np.sum(detected_gt_per_image) < detected_gt_per_image.shape[0]:
                    for annotation in np.array(anns)[detected_gt_per_image == 0]:
                        ann_cls = annotation.get_class_id()
                        conf_matrix['None'][ann_cls].append(image)
            else:  # annotations - yes, detections - no : FN
                # записать все данные по аннотациям в правый столбец None
                detection = 'None'
                for ann in anns:
                    actual_class = ann.get_class_id()
                    conf_matrix[detection][actual_class].append(image)
        else:
            if len(dets) != 0:  # annotattions - no, detections- yes : FP
                actual_class = 'None'
                for det in dets:
                    detection = det.get_class_id()
                    conf_matrix[detection][actual_class].append(image)
            else:  # annotations - no, detections - no : ????
                pass
    return conf_matrix


def convert_to_numbers(conf_matrix):
    count_dict = dict()
    for k, v in conf_matrix.items():
        count_dict[k] = dict()
        for k1, v1 in v.items():
            count_dict[k][k1] = len(conf_matrix[k][k1])
    return count_dict


def convert_confusion_matrix_to_plt_format(confusion_matrix):
    columns = list(confusion_matrix.keys())
    np_array = np.zeros(shape=(len(columns), len(columns)))
    columns.insert(0, 'class_names')

    col_names = np.array(list(confusion_matrix.keys())).reshape(-1, 1)
    for i1, (k, v) in enumerate(confusion_matrix.items()):
        for i2, (k1, v1) in enumerate(v.items()):
            np_array[i2, i1] = len(v1)
    data = np.hstack((col_names, np_array))
    return columns, data


def convert_confusion_matrix_to_plt_format_v2(confusion_matrix):
    columns = list(confusion_matrix.keys())
    data = np.zeros(shape=(len(columns), len(columns)))

    for i1, (k1, v1) in enumerate(confusion_matrix.items()):
        for i2, (k2, v2) in enumerate(v1.items()):
            data[i2, i1] = len(v2)
    return columns, data
