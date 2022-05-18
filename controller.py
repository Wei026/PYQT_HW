import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
from IMG_fuction import funtions
import os

class Mainwindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()      # 可簡寫為super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.filename = ''


    def setup_control(self):
        self.ui.open_action.triggered.connect(self.Display_img)
        self.ui.gray_action.triggered.connect(self.Gray_img)
        self.ui.save_action.triggered.connect(self.Save_img)
        self.ui.histogram_action.triggered.connect(self.Histogram_img)
        self.ui.roi_action.triggered.connect(self.roi_control)
        self.ui.Thresholding_Slider.valueChanged[int].connect(self.Threshold_control)
        self.ui.Thresholding_Button.clicked.connect(self.Thresholding)
        self.ui.Imformation_Button.clicked.connect(self.Show_imformation)

        self.ui.updown_Button.clicked.connect(self.updown_control)
        self.ui.clockwise_Button.clicked.connect(self.clockwise_control)
        self.ui.counterclockwise_Button.clicked.connect(self.counterclockwise_control)

        self.ui.erosion_action.triggered.connect(self.erosion_control)
        self.ui.dilation_action.triggered.connect(self.dilation_control)
        self.ui.opening_action.triggered.connect(self.opening_control)
        self.ui.closeing_action.triggered.connect(self.closeing_control)

    def erosion_control(self):
        # erosion_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.erode(self.img, kernel)
        funtions.label_img(self, img)


    def dilation_control(self):
        # dilation_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.dilate(self.img, kernel)
        funtions.label_img(self, img)

    def opening_control(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening_img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel, iterations=2)
        funtions.label_img(self, opening_img)

    def closeing_control(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closeing_img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel, iterations=2)
        funtions.label_img(self, closeing_img)

    def updown_control(self):
        updown_img = cv2.rotate(self.img, cv2.ROTATE_180)
        funtions.label_img(self, updown_img)

    def clockwise_control(self):
        clockwise_img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        funtions.label_img(self, clockwise_img)

    def counterclockwise_control(self):
        counterclockwise_img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        funtions.label_img(self, counterclockwise_img)

    def roi_control(self):
        self.ui.label.mousePressEvent = self.starting_point
        self.ui.label.mouseReleaseEvent = self.end_point
        self.ui.label.mouseMoveEvent = self.Mouse_move

    def starting_point(self, event):
        self.X0, self.Y0 = event.x(), event.y()

    def end_point(self, event):
        self.X1, self.Y1 = event.x(), event.y()
        self.Left_X, self.Top_Y = min(self.X0, self.X1), min(self.Y0, self.Y1)
        self.Right_X, self.Down_Y = max(self.X0, self.X1), max(self.Y0, self.Y1)
        height, weight = (self.Down_Y - self.Top_Y), (self.Right_X - self.Left_X)
        if self.filename == '':
            return 0
        else:
            if height or weight != 0:
                ROI_img = self.copyimg[int(self.Top_Y):int(self.Down_Y), int(self.Left_X):int(self.Right_X)]
                cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
                cv2.imshow("ROI", ROI_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                self.ui.statusbar.showMessage("未選取影像")

    def Mouse_move(self, point):
        self.flag = True


    def Display_img(self):
        self.ui.roi_action.setEnabled(True)
        self.filename, filetype = QFileDialog.getOpenFileName(self, "open image", "./")                                 #選擇開檔的位置
        self.img = cv2.imread(self.filename)                                                                            #讀取開檔的位置
        self.path = os.path.abspath(self.filename)                                                                      #影像資訊
        funtions.label_img(self, self.img)


    def Gray_img(self):
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)                                                      #使用cvtColor轉換格式，COLOR_BGR2GRAY是因為self.img是BGR
        funtions.label_img(self, self.gray_img)

    def Save_img(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Image', 'Image', '*.png')
        if filename == '':
            return
        cv2.imwrite(filename, self.copyimg)

    def Histogram_img(self):
        if len(self.copyimg.shape) == 3:
            color = ('b', 'g', 'r')
            alpha = (0.6, 0.6, 0.5)
            for i, col in enumerate(color):
                histr = cv2.calcHist([self.copyimg], [i], None, [256], [0, 256])                                        #(影像, 通道, 遮罩, 數值範圍)
                plt.bar(range(0, 256), histr.ravel(), color = color[i], alpha = alpha[i])
            plt.title("histogram_img")
            plt.show()
        else:
            color = 'gray'
            alpha = (0.6)
            histr = cv2.calcHist([self.copyimg], [0], None, [256], [0, 256])
            plt.figure()
            plt.bar(range(0, 256), histr.ravel(), color = color, alpha = alpha)
            plt.title("gray histogram_img")
            plt.show()

    def Thresholding(self):
        gray = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        ret, thresholding = cv2.threshold(gray, self.ui.Thresholding_Slider.value(), 255, cv2.THRESH_BINARY)            #二值化轉換
        funtions.label_img(self, thresholding)


    def Threshold_control(self):
        self.ui.Thresholding_Button.clicked.connect(self.Thresholding)
        self.ui.Thresholdvalue_label.setText(f"{self.ui.Thresholding_Slider.value()}")                                  # Thresholding_Slider.value()取得滑動條之值

    def Show_imformation(self):
        if len(self.copyimg.shape) == 3:
            height, width, channel = self.copyimg.shape
            self.ui.Information_lable.setText('影像大小: ' + str(width) + ' X ' + str(height) + '\n' + self.path + '\n' + '圖檔大小: ' + str(round(os.stat(self.path).st_size / 1e+6, 3)) + 'MB')
        else:
            height, width = self.copyimg.shape
            self.ui.Information_lable.setText('影像大小: ' + str(width) + ' X ' + str(height) + '\n' + self.path + '\n' + '圖檔大小: ' + str(round(os.stat(self.path).st_size / 1e+6, 3)) + 'MB')




if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Mainwindow_controller()
    window.show()
    sys.exit(app.exec_())