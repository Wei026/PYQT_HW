import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

def window():
   app = QApplication(sys.argv)              #app 代表「系統」的視窗程式
   widget = QWidget()                        #widger 代表一個 QWidget() 的物件 (我們開發的視窗)

   textLabel = QLabel(widget)
   textLabel.setText("fuck World!")          #存入 textLabel 這個變數中
   textLabel.move(110, 85)                   #文字與位置的設定

   widget.setGeometry(50, 50, 320, 200)      #setGeometry 可以決定視窗大小
   widget.setWindowTitle("PyQt5 Example")    #setWindowTitle 可以決定視窗名稱
   widget.show()                             #show 將視窗顯示
   sys.exit(app.exec_())                     #結束視窗應用

if __name__ == '__main__':
   window()