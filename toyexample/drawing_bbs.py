
import os

import cv2
import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, CoordinatesType

dir_imgs = 'toyexample/images'
dir_gts = 'toyexample/gts_vocpascal_format'
dir_dets = 'toyexample/dets_classname_abs_xywh'
dir_outputs = 'toyexample/images_with_bbs'

def draw_bb_into_image(image, bounding_box, color, thickness, label=None):
    if isinstance(image, str):
        image = cv2.imread(image)

    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    bb_coordinates = bounding_box.get_absolute_bounding_box(BBFormat.XYX2Y2)
    xIn = bb_coordinates[0]
    yIn = bb_coordinates[1]
    cv2.rectangle(image, (int(bb_coordinates[0]), int(bb_coordinates[1])), (int(bb_coordinates[2]), int(bb_coordinates[3])),
                  (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (xIn + thickness, yIn - th + int(12.5 * fontScale))
        xin_bb, yin_bb = int(xin_bb), int(yin_bb)
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = int(yIn + th)  # put it inside the bb
        r_Xin = int(xIn - int(thickness / 2))
        r_Yin = int(yin_bb - th - int(thickness / 2))
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image


# Drawing bb into the images
green_color = [62, 235, 59]
red_color = [255, 0, 0]
for img_file in general_utils.get_files_recursively(dir_imgs):
    # Get corresponding GT bounding boxes
    gt_annotation_file = general_utils.find_file(dir_gts, os.path.basename(img_file), match_extension=False)
    if gt_annotation_file is None:
        continue
    gt_bbs = converter.vocpascal2bb(gt_annotation_file)
    # Get corresponding detections bounding boxes
    det_annotation_file = general_utils.find_file(dir_dets, os.path.basename(img_file), match_extension=False)
    if det_annotation_file is None:
        det_bbs = []
    else:
        det_bbs = converter.text2bb(det_annotation_file, bb_type=BBType.DETECTED, bb_format=BBFormat.XYWH,type_coordinates=CoordinatesType.ABSOLUTE, img_dir=dir_imgs)
    # Leave only the annotations of cats
    gt_bbs = [bb for bb in gt_bbs if bb.get_class_id() == 'cat']
    det_bbs = [bb for bb in det_bbs if bb.get_class_id() == 'cat']
    image = cv2.imread(img_file)
    img_h, img_w, _ = image.shape
    # Draw gt bb
    for bb in gt_bbs:
        bb._y = max(bb._y, 3)
        bb._y2 = min(bb._y2, img_h-3)
        bb._x2 = min(bb._x2, img_w-3)
        image = draw_bb_into_image(image, bb, green_color, thickness=6, label=bb.get_class_id())
    for bb in det_bbs:
        print(f'{bb._image_name}: {bb._confidence}')
        print(f'{bb._image_name}')
        bb._y = max(bb._y, 3)
        bb._y2 = min(bb._y2, img_h-3)
        bb._x2 = min(bb._x2, img_w-3)
        image = draw_bb_into_image(image, bb, red_color, thickness=6, label=bb.get_class_id())
    # Uncomment to Save images
    # filename = os.path.basename(img_file)
    # cv2.imwrite(os.path.join(dir_outputs, filename), image)
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
