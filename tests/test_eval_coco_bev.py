# C:\Users\Domi\Anaconda3\envs\bev\Scripts\pip
import json
from math import isclose

from src.bounding_box import BBFormat, BBType, BoundingBox
from src.evaluators.coco_evaluator import get_coco_summary
from src.utils.converter import coco_bev2bb

# Load coco samples
gts = coco_bev2bb("tests/test_case_bev/dets", BBType.GROUND_TRUTH)
dts = coco_bev2bb("tests/test_case_bev/gts", BBType.DETECTED)
res = get_coco_summary(gts, dts)
print(gts[0])
