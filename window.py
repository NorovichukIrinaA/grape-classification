# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(958, 712)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setAcceptDrops(False)
        MainWindow.setAutoFillBackground(True)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setAnimated(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 380, 951, 271))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.tableView = QtWidgets.QTableView(self.gridLayoutWidget)
        self.tableView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.tableView.setObjectName("tableView")
        self.tableView.horizontalHeader().setCascadingSectionResizes(False)
        self.gridLayout.addWidget(self.tableView, 0, 0, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 370, 951, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(0, 350, 951, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 660, 951, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_3.addWidget(self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(530, 290, 411, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(0, 360, 941, 21))
        self.label_18.setObjectName("label_18")
        self.toolBox = QtWidgets.QToolBox(self.centralwidget)
        self.toolBox.setGeometry(QtCore.QRect(20, 20, 481, 331))
        self.toolBox.setObjectName("toolBox")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 481, 241))
        self.page_2.setObjectName("page_2")
        self.pushButton = QtWidgets.QPushButton(self.page_2)
        self.pushButton.setGeometry(QtCore.QRect(0, 180, 481, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_8 = QtWidgets.QLabel(self.page_2)
        self.label_8.setGeometry(QtCore.QRect(10, 10, 461, 61))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.page_2)
        self.label_9.setGeometry(QtCore.QRect(10, 80, 471, 81))
        self.label_9.setObjectName("label_9")
        self.toolBox.addItem(self.page_2, "")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 481, 241))
        self.page.setObjectName("page")
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_4.setGeometry(QtCore.QRect(0, 220, 481, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.frame = QtWidgets.QFrame(self.page)
        self.frame.setGeometry(QtCore.QRect(0, 0, 481, 211))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.formLayoutWidget = QtWidgets.QWidget(self.frame)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 0, 181, 211))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout_3 = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        self.label.setObjectName("label")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4)
        self.label_5 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_5)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_6)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.lineEdit_7)
        self.label_15 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_15.setObjectName("label_15")
        self.formLayout_3.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.formLayout_3.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.lineEdit_8)
        self.label_16 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_16.setObjectName("label_16")
        self.formLayout_3.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.formLayout_3.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.lineEdit_9)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.frame)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(230, 0, 201, 191))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_5 = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_5.setContentsMargins(0, 0, 0, 0)
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_20 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_20.setObjectName("label_20")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_10)
        self.label_21 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_21.setObjectName("label_21")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_11)
        self.label_22 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_22.setObjectName("label_22")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_22)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_12)
        self.label_23 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_23.setObjectName("label_23")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_23)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_13)
        self.label_24 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_24.setObjectName("label_24")
        self.formLayout_5.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.formLayout_5.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_14)
        self.label_25 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_25.setObjectName("label_25")
        self.formLayout_5.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_25)
        self.lineEdit_15 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.formLayout_5.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_15)
        self.label_26 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_26.setObjectName("label_26")
        self.formLayout_5.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.lineEdit_16 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.formLayout_5.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.lineEdit_16)
        self.label_30 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_30.setObjectName("label_30")
        self.formLayout_5.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_30)
        self.lineEdit_17 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.formLayout_5.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.lineEdit_17)
        self.toolBox.addItem(self.page, "")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setGeometry(QtCore.QRect(0, 0, 481, 241))
        self.page_3.setObjectName("page_3")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.page_3)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(0, 10, 481, 91))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_4 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_4.setContentsMargins(0, 0, 0, 0)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_13 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_13.setObjectName("label_13")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.lineEdit_20 = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_20)
        self.label_17 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_17.setObjectName("label_17")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.lineEdit_21 = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.lineEdit_21.setObjectName("lineEdit_21")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_21)
        self.pushButton_6 = QtWidgets.QPushButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.pushButton_6)
        self.toolBox.addItem(self.page_3, "")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(510, 10, 20, 341))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Классификация сорта винограда"))
        self.pushButton_3.setText(_translate("MainWindow", "очистить"))
        self.pushButton_2.setText(_translate("MainWindow", "сохранить результат"))
        self.pushButton_5.setText(_translate("MainWindow", "Прогноз"))
        self.label_18.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" text-decoration: underline;\">ДАННЫЕ</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "выбрать файл"))
        self.label_8.setText(_translate("MainWindow", "Выберете файл для анализа.\n"
"Формат файла csv\n"
"Убедитесь что данные расположены в соответствующем порядке."))
        self.label_9.setText(_translate("MainWindow", "Классификация по Al и Cr\n"
"порядок: [\'Al\', \'Cr\']\n"
"\n"
"Классификация по всем элементам\n"
"порядок: [\'Al\',\'Ba\',\'Cr\',\'Cu\',\'Fe\',\'Li\',\'Mn\',\'Na\',\'Ni\',\'Pb\',\'Rb\',\'Sr\',\'Ti\',\'Zn\',\'Ca\',\'K_\',\'Mg\']"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("MainWindow", "Выбрать фаил для анализа"))
        self.pushButton_4.setText(_translate("MainWindow", "Ввести"))
        self.label.setText(_translate("MainWindow", "Al"))
        self.label_3.setText(_translate("MainWindow", "Cr"))
        self.label_4.setText(_translate("MainWindow", "Cu"))
        self.label_5.setText(_translate("MainWindow", "Fe"))
        self.label_6.setText(_translate("MainWindow", "Li"))
        self.label_7.setText(_translate("MainWindow", "Mn"))
        self.label_15.setText(_translate("MainWindow", "Na"))
        self.label_16.setText(_translate("MainWindow", "Ni"))
        self.label_2.setText(_translate("MainWindow", "Ba"))
        self.label_20.setText(_translate("MainWindow", "Pb"))
        self.label_21.setText(_translate("MainWindow", "Rb"))
        self.label_22.setText(_translate("MainWindow", "Sr"))
        self.label_23.setText(_translate("MainWindow", "Ti"))
        self.label_24.setText(_translate("MainWindow", "Zn"))
        self.label_25.setText(_translate("MainWindow", "Ca"))
        self.label_26.setText(_translate("MainWindow", "K"))
        self.label_30.setText(_translate("MainWindow", "Mg"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), _translate("MainWindow", "Ввести все показатели"))
        self.label_13.setText(_translate("MainWindow", "Al"))
        self.label_17.setText(_translate("MainWindow", "Cu"))
        self.pushButton_6.setText(_translate("MainWindow", "Ввести"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_3), _translate("MainWindow", "Ввести Al Cu"))

        
