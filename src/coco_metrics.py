from collections import defaultdict

from .bounding_box import BBFormat


def calculate_AP_IOU(detected_bbs, groundtruth_bbs, iou_threshold=0.5):
    """Calculate the Average Precision metric as in COCO's official implementation given an IOU threshold.
    Parameters
        ----------
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;
    """
    # O cálculo do AP é feito individualmente para cada classe.
    # Ex:
    # 'cachorro' tem um AP(50) de 0.85
    # 'gato' tem um AP(50) de 0.50
    # 'coelho' tem um AP(50) de 0.70

    # group bbs by image and class
    bbs = defaultdict(lambda: {"dt": [], "gt": []})

    for det in detected_bbs:
        i_id = det.get_image_name()
        c_id = det.get_class_id()
        _bbs[i_id, c_id]["dt"].append(det)
    for gt in groundtruth_bbs:
        i_id = det.get_image_name()
        c_id = det.get_class_id()
        _bbs[i_id, c_id]["dt"].append(det)

    ious = {k: _get_ious(**_bbs[k]) for k in _bbs.keys()}

    # now we need to associate gts and dts
    matches = defaultdict(list)
    for i_id, c_id in ious.keys():
        d = ious[i_id, c_id]["dt"]
        g = ious[i_id, c_id]["gt"]
        iou = ious[i_id, c_id]["iou"]

        while len(d) > 0:
            pass

    # Get classes to be evaluated
    classes = groundtruth_bbs.get_class_id()

    bb = Bound

    ret = (
        []
    )  # list containing metrics (precision, recall, average precision) of each class

    for c in classes:

        # TODO

        values = {
            "class": c,
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None,
        }

        ret.append(values)

    return ret


def calculate_AP_scales(detected_bbs, groundtruth_bbs, min_area, max_area):
    """Calculate the Average Precision metric as in COCO's official implementation given the minimum e maximum sizes
       of the objects to be considered.
    Parameters
        ----------
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            min_area : float
                Minimum area to be considered.
            max_area : float
                Maximum area to be considered.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;
    """

    # Get classes to be evaluated
    classes = groundtruth_bbs.get_class_id()

    ret = (
        []
    )  # list containing metrics (precision, recall, average precision) of each class

    for c in classes:

        # TODO

        values = {
            "class": c,
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None,
        }

        ret.append(values)

    return ret


def calculate_AR_detections(detected_bbs, groundtruth_bbs, detections):
    """Calculate the Average Recall metric as in COCO's official implementation given the number of detections to
       be considered.
    Parameters
        ----------
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detections : float
                Number of detections per image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AR']: average recall;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;
    """

    # Get classes to be evaluated
    classes = groundtruth_bbs.get_class_id()

    ret = (
        []
    )  # list containing metrics (precision, recall, average precision) of each class

    for c in classes:

        # TODO

        values = {
            "class": c,
            "precision": None,
            "recall": None,
            "AR": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None,
        }

        ret.append(values)


def calculate_AR_scales(detected_bbs, groundtruth_bbs, min_area, max_area):
    """Calculate the Average Recall metric as in COCO's official implementation given the minimum e maximum sizes
       of the objects to be considered.
    Parameters
        ----------
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            min_area : float
                Minimum area to be considered.
            max_area : float
                Maximum area to be considered.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AR']: average recall;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;
    """

    # Get classes to be evaluated
    classes = groundtruth_bbs.get_class_id()

    ret = (
        []
    )  # list containing metrics (precision, recall, average precision) of each class

    for c in classes:

        # TODO

        values = {
            "class": c,
            "precision": None,
            "recall": None,
            "AR": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None,
        }

        ret.append(values)


def _get_ious(dt, gt):
    """ Calculate iou for every gt/detection pair """

    # sort dts by confidence level
    dt = sorted(dt, key=lambda d: s.get_confidence())

    def _jaccard(a, b):
        xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        xb, yb, x2b, y2b = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

        # innermost left x
        xi = max(xa, xb)
        # innermost right x
        x2i = min(x2a, x2b)
        # same for y
        yi = max(ya, yb)
        y2i = min(y2a, y2b)

        # calculate areas
        Aa = max(x2a - xa + 1, 0) * max(y2a - ya + 1, 0)
        Ab = max(x2b - xb + 1, 0) * max(y2b - yb + 1, 0)
        Ai = max(x2i - xi + 1, 0) * max(y2i - yi + 1, 0)
        return Ai / (Aa + Ab - Ai)

    ious = np.zeros((len(dt), len(gt)))

    for di, d in enumerate(dt):
        for gi, g in enumerate(gt):
            ious[di, gi] = _jaccard(d, g)

    return {"dt": dt, "gt": gt, "iou": ious}
