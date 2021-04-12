# Execute this script with the command below to open the GUI
# python run.py

import sys
import argparse

from PyQt5 import QtWidgets
from src.ui.run_ui import Main_Dialog
from src.ui.splash import Splash_Dialog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Open-Source Toolbox for Object Detection Metrics')
    parser.add_argument('--small', type=int, default=None, help='The lower bound for small boxes')
    parser.add_argument('--medium', type=int, default=None, help='The lower bound for medium boxes')
    parser.add_argument('--large', type=int, default=None, help='The lower bound for large boxes')
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)

    ui = Main_Dialog(coco_sizes=(('small', args.small), ('medium', args.medium), ('large', args.large)))
    ui.show()

    splash = Splash_Dialog()
    splash.show()

    sys.exit(app.exec_())
