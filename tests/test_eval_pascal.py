################################################################################
# Tests performed in the PASCAL VOC metrics
################################################################################

from math import isclose

import src.utils.converter as converter
from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from src.utils.enumerators import BBType, MethodAveragePrecision


def test_case_1():
    gts_dir = 'tests/test_case_1/gts'
    dets_dir = 'tests/test_case_1/dets'

    gts = converter.text2bb(gts_dir, BBType.GROUND_TRUTH)
    dets = converter.text2bb(dets_dir, BBType.DETECTED)

    assert (len(gts) > 0)
    assert (len(dets) > 0)

    testing_ious = [0.1, 0.3, 0.5, 0.75]

    # ELEVEN_POINT_INTERPOLATION
    expected_APs = {'object': {0.1: 0.3333333333, 0.3: 0.2683982683, 0.5: 0.0303030303, 0.75: 0.0}}
    for idx, iou in enumerate(testing_ious):
        results_dict = get_pascalvoc_metrics(
            gts, dets, iou_threshold=iou, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
        results = results_dict['per_class']
        for c, res in results.items():
            assert isclose(expected_APs[c][iou], res['AP'])

    # EVERY_POINT_INTERPOLATION
    expected_APs = {'object': {0.1: 0.3371980676, 0.3: 0.2456866804, 0.5: 0.0222222222, 0.75: 0.0}}
    for idx, iou in enumerate(testing_ious):
        results_dict = get_pascalvoc_metrics(
            gts, dets, iou_threshold=iou, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION)
        results = results_dict['per_class']
        for c, res in results.items():
            assert isclose(expected_APs[c][iou], res['AP'])
