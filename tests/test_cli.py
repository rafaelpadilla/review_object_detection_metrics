import argparse
import cli

import os
import src.utils.converter as converter
import src.utils.general_utils as general_utils
import src.utils.validations as validations
from src.utils.enumerators import BBFormat, BBType, CoordinatesType
from src.utils.enumerators import BBType, MethodAveragePrecision
from math import isclose


# Test COCO metric output:
def test_cli_coco_metric():
    tol = 1e-6

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Unused attributes assigned so that they exist in Namespace:
    args.savepath = False
    args.prgraph = False
    args.threshold = 0.5
    args.img = ''
    args.names = ''

    # Important attributes set for testing:
    args.anno_gt = 'tests/test_coco_eval/gts'
    args.anno_det = 'tests/test_coco_eval/dets'
    args.metrics = 'coco'
    args.format_gt = 'coco'
    args.format_det = 'coco'
    res = cli.__cli__(args)

    assert abs(res["AP"] - 0.503647) < tol
    assert abs(res["AP50"] - 0.696973) < tol
    assert abs(res["AP75"] - 0.571667) < tol
    assert abs(res["APsmall"] - 0.593252) < tol
    assert abs(res["APmedium"] - 0.557991) < tol
    assert abs(res["APlarge"] - 0.489363) < tol
    assert abs(res["AR1"] - 0.386813) < tol
    assert abs(res["AR10"] - 0.593680) < tol
    assert abs(res["AR100"] - 0.595353) < tol
    assert abs(res["ARsmall"] - 0.654764) < tol
    assert abs(res["ARmedium"] - 0.603130) < tol
    assert abs(res["ARlarge"] - 0.553744) < tol



def test_cli_voc_metric():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Unused attributes assigned so that they exist in Namespace:
    args.savepath = False
    args.prgraph = False
    args.threshold = 0.5
    args.img = ''
    args.names = ''
    
    # Important attributes set for testing:
    args.anno_gt = './tests/test_case_1/gts/'
    args.anno_det = './tests/test_case_1/dets/'
    args.format_gt = 'absolute'
    args.format_det = 'xywh'
    args.coord_det = 'abs'

    testing_ious = [0.1, 0.3, 0.5, 0.75]
    # ELEVEN_POINT_INTERPOLATION
    expected_APs = {'object': {0.1: 0.3333333333, 0.3: 0.2683982683, 0.5: 0.0303030303, 0.75: 0.0}}
    args.metrics='voc2007'

    for idx, iou in enumerate(testing_ious):
        args.threshold = iou
        results_dict = cli.__cli__(args)

        #results_dict = get_pascalvoc_metrics(
        #    gts, dets, iou_threshold=iou, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
        
        results = results_dict['per_class']
        for c, res in results.items():
            assert isclose(expected_APs[c][iou], res['AP'])

    # EVERY_POINT_INTERPOLATION
    expected_APs = {'object': {0.1: 0.3371980676, 0.3: 0.2456866804, 0.5: 0.0222222222, 0.75: 0.0}}
    args.metrics='voc2012'
    
    for idx, iou in enumerate(testing_ious):
        args.threshold = iou
        results_dict = cli.__cli__(args)

        #results_dict = get_pascalvoc_metrics(
        #    gts, dets, iou_threshold=iou, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION)
        
        results = results_dict['per_class']
        for c, res in results.items():
            assert isclose(expected_APs[c][iou], res['AP'])