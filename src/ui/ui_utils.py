from PyQt5 import QtCore, QtGui
import numpy as np


def image_to_pixmap(image):
    image = image.astype(np.uint8)
    if image.shape[2] == 4:
        qformat = QtGui.QImage.Format_RGBA8888
    else:
        qformat = QtGui.QImage.Format_RGB888

    image = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], qformat)
    # image= image.rgbSwapped()
    return QtGui.QPixmap(image)


def show_image_in_qt_component(image, label_component):
    pix = image_to_pixmap((image).astype(np.uint8))
    label_component.setPixmap(pix)
    label_component.setAlignment(QtCore.Qt.AlignCenter)