from collections import defaultdict
import numpy as np

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
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})

    for dt in detected_bbs:
        i_id = dt.get_image_name()
        c_id = dt.get_class_id()
        bb_info[i_id, c_id]["dt"].append(dt)
    for gt in groundtruth_bbs:
        i_id = gt.get_image_name()
        c_id = gt.get_class_id()
        bb_info[i_id, c_id]["gt"].append(gt)

    # pairwise iou for each imagexclass
    bb_info = {k: _make_iou(**bb_info[k]) for k in bb_info.keys()}

    # accumulate the scores on a per-class / iou threshold / size basis
    iou_scores = defaultdict(lambda: {"scores": [], "FP": 0, "FN": 0})

    for i_id, c_id in bb_info:
        for iou_threshold in np.linspace(0.5, 0.95, 10):
            for area_range in [None, (0, 32**2), (32**2, 96**2), (96**2, 2**31)]:
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


def _make_iou(dt, gt):
    """ Calculate iou for every gt/detection pair """

    # sort dts by confidence level
    dt = sorted(dt, key=lambda d: -d.get_confidence())

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


def _match_detections(dt, gt, iou, iou_threshold, max_dets=None, area_range=None):
    """ separate matched into detections that will depend on the iou threshold, fixed FPs and Fixed FN """

    # filter detections after max_dets
    if max_dets is not None:
        n_dets = min(max_dets, len(dt))
        dt = dt[:n_dets]

    reg_g_idx = set()
    ign_g_idx = set()

    if area_range is not None:
        # split into ignored or not ground truths
        for g_idx in range(len(gt)):
            if area_range[0] < gt[g_idx].get_area() <= area_range[1]:
                reg_g_idx.add(g_idx)
            else:
                ign_g_idx.add(g_idx)
    else:
        reg_g_idx = set(range(len(gt)))


    # hold matching dts
    matches = {}
    n_unmatched_gts = 0
    n_unmatched_dts = 0

    for d_idx, d in enumerate(dt):

        # try matching regular gt
        ggood = [g for g in reg_g_idx if g not in matches]

        if len(ggood) > 0:
            # get gt of best iou
            gbest = max(ggood, key=lambda g_idx: iou[d_idx, g_idx])
            if iou[d_idx, gbest] >= iou_threshold:
                matches[gbest] = d_idx
                continue # move on to next match

        # now try matching against ignored gts
        gbad = [g for g in ign_g_idx if g not in matches]
        if len(gbad) > 0:
            gbest = max(gbad, key=lambda g_idx: iou[d_idx, g_idx])
            if iou[d_idx, gbest] >= iou_threshold:
                matches[gbest] = d_idx
                continue # move on to next match

        # no match was possible
        # mark as unmatched if inside range
        if area_range is None or area_range[0] < d.get_area() <= area_range[1]:
            n_unmatched_dts += 1

    # count unmatched non-ignored gts
    n_unmatched_gts += len([g for g in reg_g_idx if g not in matches])
    scores = [dt[matches[g_idx]].get_confidence() for g_idx in matches if g_idx in reg_g_idx]

    return {"scores": scores, "FP": n_unmatched_dts, "FN": n_unmatched_gts}

def _ap_sweep(bb_info):
    # 
