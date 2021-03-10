

import matplotlib.pyplot as plt
import numpy as np
import src.evaluators.coco_evaluator as coco_evaluator
import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.bounding_box import BoundingBox
from src.utils.enumerators import (BBFormat, BBType, CoordinatesType,
                                   MethodAveragePrecision)

#############################################################
# DEFINE GROUNDTRUTHS AND DETECTIONS
#############################################################
dir_imgs = 'toyexample/images'
dir_gts = 'toyexample/gts_vocpascal_format'
dir_dets = 'toyexample/dets_classname_abs_xywh'
dir_outputs = 'toyexample/images_with_bbs'

def plot_bb_per_classes(dict_bbs_per_class,
                        horizontally=True,
                        rotation=0,
                        show=False,
                        extra_title=''):
    plt.close()
    if horizontally:
        ypos = np.arange(len(dict_bbs_per_class.keys()))
        plt.barh(ypos, dict_bbs_per_class.values(), align='edge')
        plt.yticks(ypos, dict_bbs_per_class.keys(), rotation=rotation)
        plt.xlabel('amount of bounding boxes')
        plt.ylabel('classes')
    else:
        plt.bar(dict_bbs_per_class.keys(), dict_bbs_per_class.values())
        plt.xlabel('classes')
        plt.ylabel('amount of bounding boxes')
    plt.xticks(rotation=rotation)
    title = f'Distribution of bounding boxes per class {extra_title}'
    plt.title(title)
    if show:
        plt.tick_params(axis='x', labelsize=10) # Set the x-axis label size
        plt.show(block=True)
    return plt

# Get annotations (ground truth and detections)
gt_bbs = converter.vocpascal2bb(dir_gts)
det_bbs = converter.text2bb(dir_dets, bb_type=BBType.DETECTED, bb_format=BBFormat.XYWH,type_coordinates=CoordinatesType.ABSOLUTE, img_dir=dir_imgs)

# Leave only the annotations of cats
gt_bbs = [bb for bb in gt_bbs if bb.get_class_id() == 'cat']
det_bbs = [bb for bb in det_bbs if bb.get_class_id() == 'cat']

# Uncomment to plot the distribution bounding boxes per classes
# dict_gt = BoundingBox.get_amount_bounding_box_all_classes(gt_bbs, reverse=False)
# plot_bb_per_classes(dict_gt, horizontally=True, rotation=0, show=True, extra_title=' (groundtruths)')
# clases_gt = [b.get_class_id() for b in gt_bbs]
# dict_det = BoundingBox.get_amount_bounding_box_all_classes(det_bbs, reverse=True)
# general_utils.plot_bb_per_classes(dict_det, horizontally=False, rotation=80, show=True, extra_title=' (detections)')

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
    dict_res = pascal_voc_evaluator.get_pascalvoc_metrics(gt_bbs, det_bbs, iou, generate_table=True, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
    voc_res = dict_res['per_class']
    pascal_voc_evaluator.plot_precision_recall_curves(voc_res, showInterpolatedPrecision=True, showAP=True)
