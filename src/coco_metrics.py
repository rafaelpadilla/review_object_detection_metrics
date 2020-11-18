""" version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py """


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for dt in detected_bbs:
        i_id = dt.get_image_name()
        c_id = dt.get_class_id()
        bb_info[i_id, c_id]["dt"].append(dt)
    for gt in groundtruth_bbs:
        i_id = gt.get_image_name()
        c_id = gt.get_class_id()
        bb_info[i_id, c_id]["gt"].append(gt)
    return bb_info

def _compute_ious(dt, gt):
    """ compute pairwise ious """

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

    return np.array([_jaccard(d, g) for d in dt] for g in gt])

def _evaluate_image(dt, gt, ious, iou_threshold, max_dets=None, area_range=None):

    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    ious = ious[dt_sort[:max_dets]]

    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] < bb.get_area() <= area_range[1])
    gt_ignore = [_is_ignore(g) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    gtm = {}
    dtm = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1-1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m>-1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = ious[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m ==-1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts
    dt_ignore = [gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in dt]

    # get score for non-ignored matched dts
    scores = [dt[d_idx].get_confidence() for d_idx in dtm if not dt_ignore[d_idx]]
    unmatched_dts = [d_idx for d_idx in range(len(dt)) if d_idx not in dtm and not dt_ignore[d_idx]]
    unmatched_gts = [g_idx for g_idx in range(len(gt)) if g_idx not in gtm and not gt_ignore[g_idx]]

    return {"scores": scores, "FP": len(unmatched_gts), "FN": len(unmatched_dts)}


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

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "FP": [], "FN": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image(**_bbs[img_id, class_id], _ious[img_id, class_id], iou_threshold)
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["FP"].append(ev["FP"])
        acc["FN"].append(ev["FN"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["FN"] = np.sum(acc["FN"])
        acc["FP"] = np.sum(acc["FP"])

    AP = {}
    # run ap_calculation per-class
    for class_id in _evals:
        scores = _evals[class_id]["scores"]
        fixed_FN = _evals[class_id]["FN"] # non-ignored non-matched gt count
        fixed_FP = _evals[class_id]["FP"] # non-ignored non-matched dt count

        n_matches = len(scores)
        n_groundtruths = n_matches + fixed_FN

        # unique thresholds, in ascending order
        thresholds, t_count = np.unique(scores, return_counts=True)
        n_thresholds = len(thresholds)

        # actually, lets use in descending order (most restrictive -> least restrictive)
        thresholds, t_count = thresholds[::-1], t_count[::-1]

        # accumulate true positives
        tp = np.cumsum(t_count) # increasing tp count (decreasing threshold)

        precision = tp / (tp + fixed_FP)
        recall = tp / n_groundtruths

        recall_thresholds = np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) + 1, endpoint=True)
        recall_indices = np.searchsorted(recall, recall_thresholds, side="right")

        interp_precisions = [np.amax(precision[r:]) if r < n_thresholds else 0.0 for r in recall_indices]

        AP[class_id] = np.mean(interp_precisions)


    # TODO: THIS does not match the requested API yet, just for testing for now...
    return AP


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
