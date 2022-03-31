from PyQt5 import QtWidgets
from controller import Mainwindow_controller

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Mainwindow_controller()
    window.show()
    sys.exit(app.exec_())