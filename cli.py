# Created by Vijay Rajagopal
import os
import sys
import argparse
import logging
from src.utils import converter
from src.utils.enumerators import (BBFormat, BBType, CoordinatesType,
                                   MethodAveragePrecision)
import src.evaluators.coco_evaluator as coco_evaluator
import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.bounding_box import BoundingBox
import matplotlib.pyplot as plt

gt_formats = ['coco', 'openimage', 'imagenet', 'abs_val', 'cvat', 'voc', 'label_me', 'YOLO']
dt_formats = ['id_ltrb', 'id_']
coords = ['abs', 'rel']

def parseArgs():
    parser = argparse.ArgumentParser()
    # Folder path to annotations
    parser.add_argument('--anno_gt', type=str)
    parser.add_argument('--anno_det', type=str)

    # Folder path to corresponding images:
    parser.add_argument('--img_gt', type=str, required=False)
    parser.add_argument('--img_det', type=str, required=False)

    # gtformat: as in the application screenshot:
    parser.add_argument('--gtformat', type=str)

    # detformat: shares input from coord (either xyrb, xywh, or coco)
    parser.add_argument('--detformat', type=str)

    # Absolute or relative (abs, rel)
    #parser.add_argument('--gtcoord',  type=str)
    parser.add_argument('--detcoord', type=str)

    # Actual computation type:
    parser.add_argument('--metrics', type=str)
    
    # metadata for metrics (not always needed)
    parser.add_argument('--names', '-n', type=str, default='')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('--imgsize', type=str, default=416)
    
    # extra data (graphs and etc.)
    parser.add_argument('--prgraph', '-pr', action='store_true')
    parser.add_argument('-sp', '--savepath', type=str, required=False, default="")
    parser.add_argument('--info', action='store_true')
    
    return parser.parse_args()

def verifyArgs(args):
    if not os.path.exists(args.anno_gt):
        raise Exception('--anno_gt path does not exist!')

    if not os.path.exists(args.anno_det):
        raise Exception('--anno_det path does not exist!')

    if args.threshold > 1 or args.threshold < 0:
        raise Exception('Incorrect range for threshold (0-1)')

    if not os.path.exists(args.savepath):
        raise Exception('Savepath does not exist!')

    if args.prgraph and args.savepath == '':
        raise Exception("Precision-Recall graph specified but no save path given!")

    if args.gtformat == 'voc' and args.names == '':
        raise Exception("VOC or ImageNet ground truth format specified, but name file not specified.")

    
    if args.img_gt == '':
        logging.warning("Image path for ground truth not specified. Assuming path is same as annotations.")
        args.img_gt = args.anno_gt

    if args.img_det == '':
        logging.warning("Image path for detection not specified. Assuming path is same as annotations.")
        args.img_det = args.anno_det

    if args.names == '':
        logging.warning("Names property empty so assuming detection format is class_id based.")

    #TODO: check if formats are legit:


def plot_coco_pr_graph(results, mAP=None, ap50=None, savePath=None, showGraphic=True):
    result = None
    plt.close()
    for classId, result in results.items():
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')

        if result['AP'] != None:
            precision = result['interpolated precision']
            recall = result['interpolated recall']
            plt.plot(recall, precision, label=f'{classId}')
        else:
            logging.warning(f"Class {classId} does not have results")
    
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    
    if mAP and ap50:
        map_str = "{0:.2f}%".format(mAP * 100)
        ap_str = "{0:.2f}%".format(ap50 * 100)
        plt.title(f'Precision x Recall curve, AP={ap_str}, AP @ 0.5={map_str}')
    else:
        plt.title('Precision x Recall curve')

    plt.legend(shadow=True)
    plt.grid()
    
    if savePath is not None:
        plt.savefig(os.path.join(savePath, 'all_classes.png'))
    if showGraphic is True:
        plt.show()
        # plt.waitforbuttonpress()
        plt.pause(0.05)


def __cli__(args):

    # check if args are correct:
    verifyArgs(args)

    # collect ground truth labels:
    if args.gtformat == 'coco':
        gt_anno = converter.coco2bb(args.anno_gt)
    elif args.gtformat == 'voc':
        gt_anno = converter.vocpascal2bb(args.anno_gt)
    elif args.gtformat == 'imagenet':
        gt_anno = converter.imagenet2bb(args.anno_gt)
    elif args.gtformat == 'labelme':
        gt_anno = converter.labelme2bb(args.anno_gt)
    elif args.gtformat == 'openimg':
        gt_anno = converter.openimage2bb(args.anno_gt, args.img_gt)
    elif args.gtformat == 'yolo':
        gt_anno = converter.yolo2bb(args.anno_gt, args.img_gt, args.names)
    elif args.gtformat == 'absolute':
        gt_anno = converter.text2bb(args.anno_gt, img_dir=args.img_gt)
    elif args.gtformat == 'cvat':
        gt_anno = converter.cvat2bb(args.anno_gt)
    else:
        raise Exception("%s is not a valid ground truth annotation format. Valid formats are: coco, voc, imagenet, labelme, openimg, yolo, absolute, cvat"%args.anno_gt)

    # collect detection truth labels:
    if args.detformat == 'coco':
        logging.warning("COCO detection format specified. Ignoring 'detcoord'...")
        # load in json:
        det_anno = converter.coco2bb(args.anno_det, bb_type=BBType.DETECTED)
    else:
        if args.detformat == 'xywh':
            # x,y,width, height
            BB_FORMAT = BBFormat.XYWH
        elif args.detformat == 'xyrb':
            # x,y,right,bottom
            BB_FORMAT = BBFormat.XYX2Y2
        else:
            raise Exception("%s is not a valid detection annotation format"%args.anno_det)
    
        if args.detcoord == 'abs':
            COORD_TYPE = CoordinatesType.ABSOLUTE
        elif args.detcoord == 'rel':
            COORD_TYPE = CoordinatesType.RELATIVE
        else:
            raise Exception("%s is not a valid detection coordinate format"%args.detcoord)
        det_anno = converter.text2bb(args.anno_det, bb_type=BBType.DETECTED, bb_format=BB_FORMAT, type_coordinates=COORD_TYPE, img_dir=args.img_det)

        # If VOC specified, then switch id based to string for detection bbox:
        #if args.gtformat == 'voc' or args.gtformat == 'imagenet':
        with open(args.names, 'r') as r:
            names = list(map(str.strip, r.readlines()))
            for det in det_anno:
                _out = names[int(det._class_id)]
                det._class_id = _out

    # print out results of annotations loaded:
    print("%d ground truth bounding boxes retrieved"%(len(gt_anno)))
    print("%d detection bounding boxes retrieved"%(len(det_anno)))

    # compute bboxes with given metric:
    if args.metrics == 'coco':
        logging.info("Running metric with COCO metric")

        # use coco_out for PR graphs and coco_sum for just the AP
        coco_sum = coco_evaluator.get_coco_summary(gt_anno, det_anno)
        coco_out = coco_evaluator.get_coco_metrics(gt_anno, det_anno, iou_threshold=args.threshold)
        
        value_only = tuple([float(_i[1]) for _i in coco_sum.items()])
        print( ('\nCOCO metric:\n'
                'AP [.5:.05:.95]: %f\n'
                'AP50: %f\n'
                'AP75: %f\n'
                'AP Small: %f\n'
                'AP Medium: %f\n'
                'AP Large: %f\n'
                'AR1: %f\n'
                'AR10: %f\n'
                'AR100: %f\n'
                'AR Small: %f\n'
                'AR Medium: %f\n'
                'AR Large: %f\n'%value_only) )

        print("Per class:")
        for item in coco_out.items():
            print("%s AP50: %f\n"%(item[0], item[1]['AP']))

        if args.prgraph:
            plot_coco_pr_graph(coco_out, mAP=coco_sum['AP50'], ap50=coco_sum['AP'], savePath=args.savepath, showGraphic=False)

    elif args.metrics == 'voc2007':
        logging.info("Running metric with VOC2007 metric")
        
        voc_sum = pascal_voc_evaluator.get_pascalvoc_metrics(gt_anno, det_anno, iou_threshold=args.threshold, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
        print("mAP: %f"%(voc_sum['mAP']))
        print("Class APs:")
        for class_item in voc_sum['per_class'].items():
            print(
                "%s: %f"%(class_item[0], class_item[1]['AP'])
            )
        
        if args.prgraph:
            pascal_voc_evaluator.plot_precision_recall_curve(voc_sum['per_class'], mAP=voc_sum['mAP'], savePath=args.savepath, showGraphic=False)

    elif args.metrics == 'voc2012' or args.metrics == 'auc':
        logging.info("Running metric with VOC2012 metric; AUC (Area Under Curve)")

        voc_sum = pascal_voc_evaluator.get_pascalvoc_metrics(gt_anno, det_anno, iou_threshold=args.threshold)
        print("mAP: %f"%(voc_sum['mAP']))
        print("Class APs:")
        for class_item in voc_sum['per_class'].items():
            print(
                "%s: %f"%(class_item[0], class_item[1]['AP'])
            )
    else:
        # Error out for incorrect metric format
        raise Exception("%s is not a valid metric (coco, voc2007, voc2012, auc)"%(args.gtformat))


if __name__ == '__main__':
    args = parseArgs()
    __cli__(args)