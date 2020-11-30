""" version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py """

from collections import defaultdict
import numpy as np

from .bounding_box import BBFormat


def coco_summary(detected_bbs, groundtruth_bbs):
    """Calculate the 12 standard metrics used in COCOEval,
        AP, AP50, AP75,
        AR_1, AR_10, AR_100,
        AP_small, AP_medium, AP_large,
        AR_small, AR_medium, AR_large,

    Parameters
        ----------
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
    Returns:
            A list of dictionaries. One dictionary for each metric.
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append(
                {
                    "class": class_id,
                    **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
                }
            )
        return res

    iou_thresholds = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    # compute simple AP with all thresholds, using all dets, and all areas
    APs = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    max_dets = [1, 10, 100]
    ARs = {
        (i, m): _evaluate(iou_threshold=i, max_dets=m, area_range=(0, np.inf))
        for i in iou_thresholds
        for m in max_dets
    }
    scales = [(0, 32 ** 2), (32 ** 2, 96 ** 2), (96 ** 2, np.inf)]
    APscales = {
        (i, s): _evaluate(iou_threshold=i, max_dets=100, area_range=s)
        for i in iou_thresholds
        for s in scales
    }

    return {"APs": APs, "ARs": ARs, "APscales": APscales}


def get_coco_metrics(
    detected_bbs,
    groundtruth_bbs,
    iou_threshold=0.5,
    area_range=(0, np.inf),
    max_dets=100,
):
    """Calculate the Average Precision metric as in COCO's official implementation given an IOU threshold.
    Parameters
        ----------
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

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
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
        )
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
        acc["NP"] = np.sum(acc["NP"])

    res = []
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        res.append(
            {
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            }
        )
    return res


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
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


def _jaccard(a, b):
    xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    xb, yb, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)
    return Ai / (Aa + Ab - Ai)


def _compute_ious(dt, gt):
    """ compute pairwise ious """

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx] = _jaccard(d, g)
    return ious


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
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = ious[d_idx, g_idx]
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
    return {"scores": scores, "matched": matched, "NP": n_gts}


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    if NP == 0:
        return None, None

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # avoid reaching outside the max recall, implicit precision eq zero
    AP = np.sum(pr[rec_idx[rec_idx < len(pr)]]) / n_recalls
    recall = rc[-1] if len(rc) else 0

    return {"AP": AP, "recall": recall}
