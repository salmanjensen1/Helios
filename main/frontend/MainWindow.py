from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(200, 200, 300, 300)
        self.setWindowTitle("My first PyQt gui")
        self.setUI()
    def setUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Hello, World!")
        self.label.move(50, 50)

        self.button1 = QtWidgets.QPushButton(self)
        self.button1.setText("Click Me, Daddy")
        self.button1.clicked.connect(self.isClicked)

    def isClicked(self):
        print("I have been clicked")
def mainWindow():
    app = QApplication(sys.argv)
    win = MyWindow()

    win.show()
    sys.exit(app.exec_())

mainWindow()