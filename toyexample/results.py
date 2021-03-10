
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import src.evaluators.coco_evaluator as coco_evaluator
import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, MethodAveragePrecision

#############################################################
# DEFINE GROUNDTRUTHS AND DETECTIONS
#############################################################
current_dir = os.path.dirname(os.path.realpath(__file__))
# dir_imgs = os.path.join(current_dir, '../data/vocpascal2012_trainval', 'JPEGImages')
# dir_imgs = os.path.join(current_dir, 'images')

# dir_dets = os.path.join(current_dir, 'dets_yolo_format')
# filepath_yolo_names = os.path.join(current_dir, '../data/vocpascal2012_trainval', 'coco.names')
#filepath_yolo_names = os.path.join(current_dir, 'voc.names')
# gt_bbs = converter.vocpascal2bb(dir_gts)
# det_bbs = converter.yolo2bb(dir_dets,  dir_imgs, filepath_yolo_names, bb_type=BBType.DETECTED)
# Leave only annotations of cats
# gt_bbs = [bb for bb in gt_bbs if bb.get_class_id() == 'cat']
# det_bbs = [bb for bb in det_bbs if bb.get_class_id() == 'cat']

dir_imgs = '/home/rafael/Downloads/testando/JPEGImages'
dir_gts = '/home/rafael/Downloads/testando/gts'
dir_dets = '/home/rafael/Downloads/testando/dets'
filepath_yolo_names = '/home/rafael/Downloads/testando/coco.names'

dir_imgs = '/home/rafael/thesis/review_object_detection_metrics/toyexample/images'
dir_gts = '/home/rafael/thesis/review_object_detection_metrics/toyexample/gts_vocpascal_format'
dir_dets = '/home/rafael/thesis/review_object_detection_metrics/toyexample/dets_yolo_format'
filepath_yolo_names = '/home/rafael/thesis/review_object_detection_metrics/toyexample/voc.names'


# gt_bbs = converter.vocpascal2bb(dir_gts)
# det_bbs = converter.yolo2bb(dir_dets,  dir_imgs, filepath_yolo_names, bb_type=BBType.DETECTED)

# gt_bbs = [det for det in gt_bbs if det.get_class_id() == 'cat']
# det_bbs = [det for det in det_bbs if det.get_class_id() == 'cat']

gt_bbs = pickle.load(open('gts.pickle', 'rb'))
det_bbs = pickle.load(open('dets.pickle', 'rb'))

# dict_bbs_per_class = BoundingBox.get_amount_bounding_box_all_classes(gt_bbs, reverse=True)
# general_utils.plot_bb_per_classes(dict_bbs_per_class, horizontally=False, rotation=90, show=True, extra_title=' (groundtruths)')

# dict_bbs_per_class = BoundingBox.get_amount_bounding_box_all_classes(det_bbs, reverse=True)
# general_utils.plot_bb_per_classes(dict_bbs_per_class, horizontally=False, rotation=90, show=True, extra_title=' (detections)')

#############################################################
# EVALUATE WITH COCO METRICS
#############################################################
coco_res1 = coco_evaluator.get_coco_summary(gt_bbs, det_bbs)
coco_res2 = coco_evaluator.get_coco_metrics(gt_bbs, det_bbs)
#############################################################
# EVALUATE WITH VOC PASCAL METRICS
#############################################################
ious = [0.5, 0.75]
voc_res = {}
for iou in ious:
    voc_res[iou], mAP = pascal_voc_evaluator.get_pascalvoc_metrics(gt_bbs, det_bbs, iou, generate_table=True, method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION)
    pascal_voc_evaluator.plot_precision_recall_curves(voc_res[iou], showInterpolatedPrecision=True, showAP=True)
