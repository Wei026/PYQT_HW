import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
import os

class funtions():
    def __init__(self):
        super().__init__()      # 可簡寫為super().__init__()

    def label_img(self, img):
        if len(img.shape) == 3:
            self.copyimg = img                                                                                          # 將讀到的圖片複製一份到copyimg裡面
            height, width, channel = img.shape                                                                          # 讀取圖片的 shape
            btyesPerline = 3 * width                                                                                    # RGB是三個通道
            self.qimg = QImage(img, width, height, btyesPerline,QImage.Format_RGB888).rgbSwapped()                      # 轉成 OpenCV (numpy) 的格式圖片轉換成 QImage 的格式
            self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))                                                       # 將圖片在 label 中顯示
        elif len(img.shape) == 2:
            self.copyimg = img                                                                                          #將讀到的圖片複製一份到copyimg裡面
            height, width = img.shape
            gray_bytesPerline = 1 * width                                                                               # 灰階只有1個通道
            self.gray_qimg = QImage(img, width, height, gray_bytesPerline, QImage.Format_Indexed8)
            self.ui.label.setPixmap(QPixmap.fromImage(self.gray_qimg))
