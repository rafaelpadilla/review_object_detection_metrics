""" version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Notes:
        1) The default area thresholds here follows the values defined in COCO, that is,
        small:           area <= 32**2
        medium: 32**2 <= area <= 96**2
        large:  96**2 <= area.
        If area is not specified, all areas are considered.

        2) COCO's ground truths contain an 'area' attribute that is associated with the segmented area if
        segmentation-level information exists. While coco uses this 'area' attribute to distinguish between
        'small', 'medium', and 'large' objects, this implementation simply uses the associated bounding box
        area to filter the ground truths.

        3) COCO uses floating point bounding boxes, thus, the calculation of the box area
        for IoU purposes is the simple open-ended delta (x2 - x1) * (y2 - y1).
        PASCALVOC uses integer-based bounding boxes, and the area includes the outer edge,
        that is, (x2 - x1 + 1) * (y2 - y1 + 1). This implementation assumes the open-ended (former)
        convention for area calculation.
"""
from src.bounding_box_rotated import BoundingBoxRotated
from collections import defaultdict

import numpy as np
from src.bounding_box import BBFormat, BoundingBox


def get_nuscenes_summary(groundtruth_bbs, detected_bbs):
    """
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
    Returns:
            A dictionary with one entry for each metric.
    """
    # separate bbs per image X class
    # ROTATION INVARIANT
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise dists
    _dists = {k: _compute_dists(**v) for k, v in _bbs.items()}

    def _evaluate(dist_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(
            lambda: {
                "scores": [],
                "matched": [],
                "NP": [],
                "ATE": [],
                "AOE": [],
                "ASE": [],
            }
        )
        for img_id, class_id in _bbs:
            # ROTATION ADJUSTED
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _dists[img_id, class_id],
                dist_threshold,
                max_dets,
                area_range,
                calc_ate=True,
                calc_aoe=True,
                calc_ase=True,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])
            acc["ATE"].append(ev["ATE"])
            acc["AOE"].append(ev["AOE"])
            acc["ASE"].append(ev["ASE"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
            acc["NP"] = np.sum(acc["NP"])
            acc["ATE"] = np.mean(np.concatenate(acc["ATE"]))
            acc["AOE"] = np.mean(np.concatenate(acc["AOE"]))
            acc["ASE"] = np.mean(np.concatenate(acc["ASE"]))
        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append(
                {
                    "class": class_id,
                    "ATE": ev["ATE"],
                    "AOE": ev["AOE"],
                    "ASE": ev["ASE"],
                    # ROTATION INVARIANT
                    **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
                }
            )
        return res

    dist_thresholds = np.array([0.5, 1.0, 2.0, 4.0])

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        # ROTATION ADJUSTED
        i: _evaluate(dist_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in dist_thresholds
    }

    AP_05m = np.mean([x["AP"] for x in full[0.50] if x["AP"] is not None])
    AP_1m = np.mean([x["AP"] for x in full[1.0] if x["AP"] is not None])
    AP_2m = np.mean([x["AP"] for x in full[2.0] if x["AP"] is not None])
    AP_4m = np.mean([x["AP"] for x in full[4.0] if x["AP"] is not None])
    AP = np.mean([x["AP"] for k in full for x in full[k] if x["AP"] is not None])
    tp_metrics_distance = 2.0
    ATE = np.mean([x["ATE"] for x in full[tp_metrics_distance] if x["ATE"] is not None])
    AOE = np.mean([x["AOE"] for x in full[tp_metrics_distance] if x["AOE"] is not None])
    ASE = np.mean([x["ASE"] for x in full[tp_metrics_distance] if x["ASE"] is not None])
    # max recall for 100 dets can also be calculated here
    AR100 = np.mean(
        [
            x["TP"] / x["total positives"]
            for k in full
            for x in full[k]
            if x["TP"] is not None
        ]
    )

    small = {
        # ROTATION ADJUSTED
        i: _evaluate(dist_threshold=i, max_dets=100, area_range=(0, 32**2))
        for i in dist_thresholds
    }
    APsmall = [x["AP"] for k in small for x in small[k] if x["AP"] is not None]
    APsmall = np.nan if APsmall == [] else np.mean(APsmall)
    ARsmall = [
        x["TP"] / x["total positives"]
        for k in small
        for x in small[k]
        if x["TP"] is not None
    ]
    ARsmall = np.nan if ARsmall == [] else np.mean(ARsmall)

    medium = {
        # ROTATION ADJUSTED
        i: _evaluate(dist_threshold=i, max_dets=100, area_range=(32**2, 96**2))
        for i in dist_thresholds
    }
    APmedium = [x["AP"] for k in medium for x in medium[k] if x["AP"] is not None]
    APmedium = np.nan if APmedium == [] else np.mean(APmedium)
    ARmedium = [
        x["TP"] / x["total positives"]
        for k in medium
        for x in medium[k]
        if x["TP"] is not None
    ]
    ARmedium = np.nan if ARmedium == [] else np.mean(ARmedium)

    large = {
        # ROTATION ADJUSTED
        i: _evaluate(dist_threshold=i, max_dets=100, area_range=(96**2, np.inf))
        for i in dist_thresholds
    }
    APlarge = [x["AP"] for k in large for x in large[k] if x["AP"] is not None]
    APlarge = np.nan if APlarge == [] else np.mean(APlarge)
    ARlarge = [
        x["TP"] / x["total positives"]
        for k in large
        for x in large[k]
        if x["TP"] is not None
    ]
    ARlarge = np.nan if ARlarge == [] else np.mean(ARlarge)

    max_det1 = {
        # ROTATION ADJUSTED
        i: _evaluate(dist_threshold=i, max_dets=1, area_range=(0, np.inf))
        for i in dist_thresholds
    }
    AR1 = np.mean(
        [
            x["TP"] / x["total positives"]
            for k in max_det1
            for x in max_det1[k]
            if x["TP"] is not None
        ]
    )

    max_det10 = {
        # ROTATION ADJUSTED
        i: _evaluate(dist_threshold=i, max_dets=10, area_range=(0, np.inf))
        for i in dist_thresholds
    }
    AR10 = np.mean(
        [
            x["TP"] / x["total positives"]
            for k in max_det10
            for x in max_det10[k]
            if x["TP"] is not None
        ]
    )

    return {
        "AP": AP,
        "AP_05m": AP_05m,
        "AP_1m": AP_1m,
        "AP_2m": AP_2m,
        "AP_4m": AP_4m,
        "ATE": ATE,
        "AOE": AOE,
        "ASE": ASE,
        "APsmall": APsmall,
        "APmedium": APmedium,
        "APlarge": APlarge,
        "AR1": AR1,
        "AR10": AR10,
        "AR100": AR100,
        "ARsmall": ARsmall,
        "ARmedium": ARmedium,
        "ARlarge": ARlarge,
    }


def _group_detections(dt, gt):
    """simply group gts and dts on a imageXclass basis"""
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        c_id = d.get_class_id()
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        c_id = g.get_class_id()
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(a: BoundingBoxRotated | BoundingBox):
    """COCO does not consider the outer edge as included in the bbox"""
    if isinstance(a, BoundingBoxRotated):
        return a.get_area()
    else:
        x, y, x2, y2 = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        return (x2 - x) * (y2 - y)


def _compute_dists(dt, gt):
    """compute pairwise ious"""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    if isinstance(dt[0], BoundingBoxRotated) and isinstance(gt[0], BoundingBoxRotated):
        dists = np.zeros((len(dt), len(gt)))
        for g_idx, g in enumerate(gt):
            for d_idx, d in enumerate(dt):
                dists[d_idx, g_idx] = BoundingBoxRotated.center_distance(d, g)
        return dists
    else:
        raise ValueError("dt and gt must be of the same type")


def _evaluate_image(
    dt: list[BoundingBoxRotated] | list[BoundingBox],
    gt: list[BoundingBoxRotated] | list[BoundingBox],
    dists,
    dist_threshold,
    max_dets=None,
    area_range=None,
    calc_ate=False,
    calc_aoe=False,
    calc_ase=False,
):
    """use COCO's method to associate detections to ground truths
    Args:
        dt: list of detections
        gt: list of ground truths
        dists: pairwise distance matrix
        dist_threshold: distance threshold
        max_dets: maximum number of detections to consider
        area_range: area range to consider
        calc_ate: whether to calculate ATE (average translation error)
        calc_aoe: whether to calculate AOE (average orientation error)
        calc_ase: whether to calculate ASE (average scale error)
    """
    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]

    dists = dists[dt_sort[:max_dets]]

    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        # ROTATION ADJUSTED
        return not (area_range[0] <= _get_area(bb) <= area_range[1])

    # ROTATION ADJUSTED
    gt_ignore = [_is_ignore(g) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    dists = dists[:, gt_sort]

    gtm = {}
    dtm = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        min_dist_to_prev_gt = min(dist_threshold, 1000)  # in meters
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop ( after this there will only be ignores)
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if dists[d_idx, g_idx] > min_dist_to_prev_gt:
                continue
            # if match successful and best so far, store appropriately
            min_dist_to_prev_gt = dists[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d)
        for d_idx, d in enumerate(dt)
    ]

    # get score for non-ignored dts
    scores = [
        dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]
    ]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return_val = {
        "scores": scores,
        "matched": matched,
        "NP": n_gts,
    }
    if calc_aoe:
        aoe = [
            BoundingBoxRotated.orientation_error(dt[gtm[g_idx]], gt[g_idx])
            for g_idx in gtm
        ]
        return_val["AOE"] = aoe
    if calc_ate:
        ate = [
            BoundingBoxRotated.translation_error(dt[gtm[g_idx]], gt[g_idx])
            for g_idx in gtm
        ]
        return_val["ATE"] = ate
    if calc_ase:
        ase = [
            BoundingBoxRotated.scale_error(dt[gtm[g_idx]], gt[g_idx]) for g_idx in gtm
        ]
        return_val["ASE"] = ase
    return return_val


def _compute_ap_recall(
    scores,
    matched,
    NP,
    recall_thresholds=None,
):
    """This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced.
    """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None,
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    # matched is list of bools
    matched = matched[inds]
    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)
    # NP = number of positives gts
    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
    }
