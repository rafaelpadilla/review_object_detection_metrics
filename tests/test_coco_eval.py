import json
from math import isclose

from src.bounding_box import BBFormat, BBType, BoundingBox
from src.evaluators.coco_evaluator import get_coco_summary
from src.utils.converter import coco2bb

# Load coco samples
gts = coco2bb('tests/test_coco_eval/gts', BBType.GROUND_TRUTH)
dts = coco2bb('tests/test_coco_eval/dets', BBType.DETECTED)

res = get_coco_summary(gts, dts)

# Compare results to those obtained with coco's official implementation
tol = 1e-6

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
