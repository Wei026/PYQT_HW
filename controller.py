import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
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
            if height or weight !=0:
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
        self.copyimg = self.img                                                                                         #將讀到的圖片複製一份到copyimg裡面
        height, width, channel = self.img.shape                                                                         #讀取圖片的 shape
        btyesPerline = 3 * width                                                                                        #RGB是三個通道
        self.qimg = QImage(self.img, width, height, btyesPerline, QImage.Format_RGB888).rgbSwapped()                    #轉成 OpenCV (numpy) 的格式圖片轉換成 QImage 的格式
        self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))                                                           #將圖片在 label 中顯示

    def Gray_img(self):
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)                                                      #使用cvtColor轉換格式，COLOR_BGR2GRAY是因為self.img是BGR
        self.copyimg = self.gray_img                                                                                    ##將讀到的圖片複製一份到copyimg裡面
        height, width = self.gray_img.shape
        gray_bytesPerline = 1 * width                                                                                   #灰階只有1個通道
        self.gray_qimg = QImage(self.gray_img, width, height, gray_bytesPerline, QImage.Format_Indexed8)
        self.ui.label.setPixmap(QPixmap.fromImage(self.gray_qimg))

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
        gray = cv2.cvtColor(self.copyimg, cv2.COLOR_BGR2GRAY)
        ret, thresholding = cv2.threshold(gray, self.ui.Thresholding_Slider.value(), 255, cv2.THRESH_BINARY)
        height, width = gray.shape
        bytesPerline = 1 * width
        self.qImg = QImage(thresholding, width, height, bytesPerline, QImage.Format_Grayscale8).rgbSwapped()            #灰階只有1個通道
        self.qpixmap = QPixmap.fromImage(self.qImg)
        self.ui.label.setPixmap(QPixmap.fromImage(self.qImg))


    def Threshold_control(self):
        self.ui.Thresholding_Button.clicked.connect(self.Thresholding)
        self.ui.Thresholdvalue_label.setText(f"{self.ui.Thresholding_Slider.value()}")                                  # Thresholding_Slider.value()取得滑動條之值

    def Show_imformation(self):
        height, width, channel = self.copyimg.shape
        self.ui.Information_lable.setText('影像大小: ' + str(width) + ' X ' + str(height) + '\n' + self.path + '\n' + '圖檔大小: ' + str(round(os.stat(self.path).st_size / 1e+6, 3)) + 'MB')



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Mainwindow_controller()
    window.show()
    sys.exit(app.exec_())