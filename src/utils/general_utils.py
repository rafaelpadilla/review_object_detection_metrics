import fnmatch
import os

import cv2
import numpy as np


def get_files_recursively(directory, extension="*"):
    if '.' not in extension:
        extension = '*.' + extension
    files = [
        os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(directory)
        for f in fnmatch.filter(files, extension)
    ]
    return files


# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convert_to_relative_values(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # YOLO's format
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convert_to_absolute_values(size, box):
    w_box = size[0] * box[2]
    h_box = size[1] * box[3]

    x1 = (float(box[0]) * float(size[0])) - (w_box / 2)
    y1 = (float(box[1]) * float(size[1])) - (h_box / 2)
    x2 = x1 + w_box
    y2 = y1 + h_box
    return (round(x1), round(y1), round(x2), round(y2))


def add_bb_into_image(image, bb, color=(255, 0, 0), thickness=2, label=None):
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    x1, y1, x2, y2 = bb.get_absolute_bounding_box(BBFormat.XYX2Y2)
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (x1 + thickness, y1 - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = y1 + th  # put it inside the bb
        r_Xin = x1 - int(thickness / 2)
        r_Yin = y1 - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image


def remove_file_extension(filename):
    return os.path.join(os.path.dirname(filename), os.path.splitext(filename)[0])


def get_files_dir(directory, extensions=['*']):
    ret = []
    for extension in extensions:
        if extension == '*':
            ret += [f for f in os.listdir(directory)]
            continue
        elif extension is None:
            # accepts all extensions
            extension = ''
        elif '.' not in extension:
            extension = f'.{extension}'
        ret += [f for f in os.listdir(directory) if f.endswith(extension)]
    return ret


def get_files_recursively(directory, extension="*"):
    if '.' not in extension:
        extension = '*.' + extension
    files = [
        os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(directory)
        for f in fnmatch.filter(files, extension)
    ]
    return files


def find_file(directory, file_name, match_extension=True):
    for dirpath, dirnames, files in os.walk(directory):
        for f in files:
            f1 = os.path.basename(f)
            f2 = file_name
            if not match_extension:
                f1 = os.path.splitext(f1)[0]
                f2 = os.path.splitext(f2)[0]
            if f1 == f2:
                return os.path.join(dirpath, os.path.basename(f))
    return None


def get_image_resolution(image_file):
    if image_file is None or not os.path.isfile(image_file):
        print(f'Warning: Path {image_file} not found.')
        return None
    img = cv2.imread(image_file)
    if img is None:
        print(f'Warning: Error loading the image {image_file}.')
        return None
    h, w, _ = img.shape
    return {'height': h, 'width': w}


def draw_bb_into_image(image, boundingBox, color, thickness, label=None):
    if isinstance(image, str):
        image = cv2.imread(image)

    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    xIn = boundingBox[0]
    yIn = boundingBox[1]
    cv2.rectangle(image, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]),
                  (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (xIn + thickness, yIn - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = yIn + th  # put it inside the bb
        r_Xin = xIn - int(thickness / 2)
        r_Yin = yin_bb - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image
