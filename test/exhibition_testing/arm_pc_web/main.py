#!/usr/bin/env python3
# encoding: utf-8

import os
import re
import cv2
import sys
import copy
import math
import time
import sqlite3
import threading
from functools import partial
import resource_rc
from socket import * 
from servo_controller import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtSql import QSqlDatabase, QSqlQuery


DOF = 6
if 'ROBOT_DOF' in os.environ:
    DOF = int(os.environ['ROBOT_DOF'])

if DOF == 7:
    from ui_7dof import Ui_Form
else:
    from ui_6dof import Ui_Form


class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.set_window_position()
        
        self.path = os.path.split(os.path.realpath(__file__))[0]
        self.actdir = os.path.join(self.path, "ActionGroups")
        self.setWindowIcon(QIcon(os.path.join(self.path, 'resources/arm.png')))
        self.tabWidget.setCurrentIndex(0)  # 设置默认标签为第一个标签(set the default label as the first one)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置选中整行，若不设置默认选中单元格(Set to select the entire row, if not, select the default cell)
        self.message = QMessageBox()
        if DOF == 6:
            self.joints = [1, 2, 3, 4, 5, 10]
        if DOF == 7:
            self.joints = [1, 2, 3, 4, 5, 6, 10]
        self.min_time = 500
        self.button_controlaction_clicked('refresh')

        ########################主界面(main interface)###############################
        self.lineEdit_object = {}
        self.validator = QIntValidator(0, 1000)
        for j in self.joints:
            obj = self.findChild(QLineEdit, "lineEdit_%d"  % j)
            if obj:
                obj.setValidator(self.validator)
                self.lineEdit_object[j] = obj

        # 滑竿同步对应文本框的数值,及滑竿控制相应舵机转动与valuechange函数绑定(Synchronize the value of the corresponding text box with the slider, and bind the slider to the 'valuechange' function to control the rotation of the corresponding servo)
        self.current_positions = {}
        self.horizontalSlider_pulse_object = {}
        for j in self.joints:
            obj = self.findChild(QSlider, "horizontalSlider_pulse_%d" % j) 
            if obj:
                obj.valueChanged.connect(partial(self.valuechange1, j))
                self.horizontalSlider_pulse_object[j] = obj
                self.current_positions[j] = obj.value()

        self.horizontalSlider_dev_object = {}
        for j in self.joints:
            obj = self.findChild(QSlider, "horizontalSlider_dev_%d" % j)
            if obj:
                obj.valueChanged.connect(partial(self.valuechange2, j))
                self.horizontalSlider_dev_object[j] = obj
       
        self.label_object = {}
        for j in self.joints:
            obj = self.findChild(QLabel, "label_d%d" % j)
            if obj:
                self.label_object[j] = obj

        self.radioButton_zn.toggled.connect(lambda: self.language(self.radioButton_zn))
        self.radioButton_en.toggled.connect(lambda: self.language(self.radioButton_en))        
        self.chinese = True
        try:
            if 'zh_CN' in os.environ['LANG']:
                self.radioButton_zn.setChecked(True)
            else:
                self.chinese = False
                self.radioButton_en.setChecked(True)
        except:
            self.radioButton_zn.setChecked(True)
        
        # tableWidget点击获取定位的信号与icon_position函数（添加运行图标）绑定(Bind the signal that obtains the positioning when the tableWidget is clicked to the 'icon_position' function (add the running icon))
        self.tableWidget.pressed.connect(self.icon_position)

        self.lineEdit_time.setValidator(QIntValidator(20, 30000))

        # 将编辑动作组的按钮点击时的信号与button_editaction_clicked函数绑定(Bind the signal when the button for editing the action group is clicked to the 'button_editaction_clicked' function)
        self.Button_ServoPowerDown.pressed.connect(lambda: self.button_editaction_clicked('servoPowerDown'))
        self.Button_AngularReadback.pressed.connect(lambda: self.button_editaction_clicked('angularReadback'))
        self.Button_AddAction.pressed.connect(lambda: self.button_editaction_clicked('addAction'))
        self.Button_DelectAction.pressed.connect(lambda: self.button_editaction_clicked('delectAction'))
        self.Button_DelectAllAction.pressed.connect(lambda: self.button_editaction_clicked('delectAllAction'))                                                 
        self.Button_UpdateAction.pressed.connect(lambda: self.button_editaction_clicked('updateAction'))
        self.Button_InsertAction.pressed.connect(lambda: self.button_editaction_clicked('insertAction'))
        self.Button_MoveUpAction.pressed.connect(lambda: self.button_editaction_clicked('moveUpAction'))
        self.Button_MoveDownAction.pressed.connect(lambda: self.button_editaction_clicked('moveDownAction'))        

        # 将运行及停止运行按钮点击的信号与button_runonline函数绑定(Bind the signal when the running and stop-running buttons are clicked to the 'button_runonline' function)
        self.Button_Run.clicked.connect(lambda: self.button_run('run'))

        self.Button_OpenActionGroup.pressed.connect(lambda: self.button_flie_operate('openActionGroup'))
        self.Button_SaveActionGroup.pressed.connect(lambda: self.button_flie_operate('saveActionGroup'))
        self.Button_ReadDeviation.pressed.connect(lambda: self.button_flie_operate('readDeviation'))
        self.Button_DownloadDeviation.pressed.connect(lambda: self.button_flie_operate('downloadDeviation'))
        self.Button_TandemActionGroup.pressed.connect(lambda: self.button_flie_operate('tandemActionGroup'))
        self.Button_ReSetServos.pressed.connect(lambda: self.button_re_clicked('reSetServos'))
        
        # 将控制动作的按钮点击的信号与action_control_clicked函数绑定(Bind the signal when the action control button is clicked to the 'action_control_clicked' function)
        self.Button_DelectSingle.pressed.connect(lambda: self.button_controlaction_clicked('delectSingle'))
        self.Button_AllDelect.pressed.connect(lambda: self.button_controlaction_clicked('allDelect'))
        self.Button_RunAction.pressed.connect(lambda: self.button_controlaction_clicked('runAction'))
        self.Button_StopAction.pressed.connect(lambda: self.button_controlaction_clicked('stopAction'))
        self.Button_Refresh.pressed.connect(lambda: self.button_controlaction_clicked('refresh'))
        self.Button_Quit.pressed.connect(lambda: self.button_controlaction_clicked('quit'))

        for key, value in self.horizontalSlider_dev_object.items():
            value.setEnabled(False)

        

        self.devNew = [0, 0, 0, 0, 0]
        self.dev_change = False 
        self.resetServos_ = False
        self.readDevOk = False
        self.totalTime = 0
        self.row = 0
        self.start_run = True
        self.use_time_list = []
        self.use_time_list_ = []
        
        self.readOrNot = False
        self.running = False
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_position)
        self.update_timer.start(25)


    def update_position(self):
        if self.running or self.resetServos_:
            return
        for j in self.joints:
            p = self.horizontalSlider_pulse_object[j].value()
            if abs(p - self.current_positions[j]) > 0.01:
                p = self.current_positions[j] * 0.8 + 0.2 * p
                setServoPulse(j, int(p), 20)
                self.current_positions[j] = p


    def set_window_position(self):
        # 窗口居中(center the window)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def language(self, name):
        if self.radioButton_zn.isChecked() and name.text() == '中文':
            self.chinese = True
            self.Button_ServoPowerDown.setText("手掰编程")
            self.Button_AngularReadback.setText("角度回读")
            self.Button_AddAction.setText("添加动作")
            self.Button_DelectAction.setText("删除动作")
            self.Button_UpdateAction.setText("更新动作")
            self.Button_InsertAction.setText("插入动作")
            self.Button_MoveUpAction.setText("上移动作")
            self.Button_MoveDownAction.setText("下移动作")        
            self.Button_OpenActionGroup.setText("打开动作文件")
            self.Button_SaveActionGroup.setText("保存动作文件")
            self.Button_ReadDeviation.setText("读取偏差")
            self.Button_DownloadDeviation.setText("下载偏差")
            self.Button_TandemActionGroup.setText("串联动作文件")
            self.Button_ReSetServos.setText("舵机回中")
            self.Button_DelectSingle.setText("单个擦除")
            self.Button_AllDelect.setText("全部擦除")
            self.Button_RunAction.setText("动作运行")
            self.Button_StopAction.setText("动作停止")
            self.Button_Run.setText("运行")
            self.checkBox.setText("循环")
            self.label_action.setText("动作组:")
            self.label_time.setText("时间")
            self.label_time_2.setText("运行总时间")
            self.Button_Quit.setText("退出")
            self.Button_DelectAllAction.setText("删除全部")
            self.Button_Refresh.setText("刷新")
            self.label_open.setText("开")
            self.label_close.setText("合")

            for i in range(1, 10):
                obj_up = self.findChild(QLabel, "label_up_%d" % i)
                obj_down = self.findChild(QLabel, "label_down_%d" % i)
                obj_left = self.findChild(QLabel, "label_left_%d" % i)
                obj_right = self.findChild(QLabel, "label_right_%d" % i)
                if obj_up:
                    obj_up.setText("上")
                if obj_down:
                    obj_down.setText("下")
                if obj_left:
                    obj_left.setText("左")
                if obj_right:
                    obj_right.setText("右")

            item = self.tableWidget.horizontalHeaderItem(1)
            item.setText("编号")
            item = self.tableWidget.horizontalHeaderItem(2)
            item.setText("时间")
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), "普通模式")

        elif self.radioButton_en.isChecked() and name.text() == 'English':
            self.chinese = False
            self.Button_ServoPowerDown.setText("Manual")
            self.Button_AngularReadback.setText("Read angle")
            self.Button_AddAction.setText("Add action")
            self.Button_DelectAction.setText("Delete action")
            self.Button_UpdateAction.setText("Update action")
            self.Button_InsertAction.setText("Insert action")
            self.Button_MoveUpAction.setText("Action upward")
            self.Button_MoveDownAction.setText("Action down")        
            self.Button_OpenActionGroup.setText("Open action file")
            self.Button_SaveActionGroup.setText("Save action file")
            self.Button_ReadDeviation.setText("Read deviation")
            self.Button_DownloadDeviation.setText("Download deviation")
            self.Button_TandemActionGroup.setText("Integrate file")
            self.Button_ReSetServos.setText("Reset servo")
            self.Button_DelectSingle.setText("Erase single")
            self.Button_AllDelect.setText("All erase")
            self.Button_RunAction.setText("Run action")
            self.Button_StopAction.setText("Stop")
            self.Button_Run.setText("Run")           
            self.checkBox.setText("Loop")
            self.label_action.setText("Action group:")
            self.label_time.setText("Duration")
            self.label_time_2.setText("Total duration")  
            self.Button_Quit.setText("Quit")
            self.Button_DelectAllAction.setText("Delete all")
            self.Button_Refresh.setText("Refresh")
            self.label_open.setText("Open")
            self.label_close.setText("Close")
            for i in range(1, 10):
                obj_up = self.findChild(QLabel, "label_up_%d" % i)
                obj_down = self.findChild(QLabel, "label_down_%d" % i)
                obj_left = self.findChild(QLabel, "label_left_%d" % i)
                obj_right = self.findChild(QLabel, "label_right_%d" % i)
                if obj_up:
                    obj_up.setText("Up")
                if obj_down:
                    obj_down.setText("Down")
                if obj_left:
                    obj_left.setText("Left")
                if obj_right:
                    obj_right.setText("Right")
            item = self.tableWidget.horizontalHeaderItem(1)
            item.setText("Index")
            item = self.tableWidget.horizontalHeaderItem(2)
            item.setText("Time")
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), "Normal Mode")

    # 弹窗提示函数(pop-up prompt function)
    def message_from(self, str):
        try:
            QMessageBox.about(self, '', str)
            time.sleep(0.01)
        except:
            pass
    
    def message_From(self, str):
        self.message_from(str)
   
    # 弹窗提示函数(pop-up prompt function)
    def message_delect(self, str):
        messageBox = QMessageBox()
        messageBox.setWindowTitle(' ')
        messageBox.setText(str)
        messageBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        messageBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)
        return messageBox.exec_()

    # 窗口退出(exit the window)
    def closeEvent(self, e):        
        result = QMessageBox.question(self,
                                    "关闭窗口提醒",
                                    "exit?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if result == QMessageBox.Yes:
            self.camera_ui = True
            self.camera_ui_break = True
            QWidget.closeEvent(self, e)
        else:
            e.ignore()
    
    def keyPressEvent(self, event):
        if event.key() == 16777220:
            self.resetServos_ = True
            for i in self.joints:
                pulse = int(self.lineEdit_object[i].text())
                self.horizontalSlider_pulse_object[i].setValue(pulse)
                self.current_positions[i] = pulse
                setServoPulse(i, pulse, 500)
            self.resetServos_ = False
    
    def tabchange(self):
        if self.tabWidget.currentIndex() == 1:
            if self.chinese:
                self.message_From('使用此面板时，请确保只连接了一个舵机，否则会引起冲突！')
            else:
                self.message_From('Before debugging servo,make sure that the servo controller is connected with ONE servo.Otherwise it may cause a conflict!')
        
    
    # 滑竿同步对应文本框的数值,及滑竿控制相应舵机转动(synchronize the value of the corresponding text box with the slider, and use the slider to control the rotation of the corresponding servo)
    def valuechange1(self, name, obj):
        if not self.resetServos_:
            servo_pulse = self.horizontalSlider_pulse_object[name].value()
            #setServoPulse(name, servo_pulse, self.min_time)
            self.lineEdit_object[name].setText(str(servo_pulse))    

    def valuechange2(self, name):
        if self.readDevOk:
            self.devNew[0] = self.horizontalSlider_dev_object[name].value()
            setServoDeviation(name, self.devNew[0])
            self.label_object[name].setText(str(self.devNew[0]))
            if self.devNew[0] < 0:
                self.devNew[0] = 0xff + self.devNew[0] + 1 
        else:
            self.message_From('请先读取偏差!')
                     
    # 复位按钮点击事件(reset button click event)
    def button_re_clicked(self, name):
        self.resetServos_ = True
        if name == 'reSetServos':
            for i in self.joints:
                self.horizontalSlider_pulse_object[i].setValue(500)
                self.current_positions[i] = 500
                self.lineEdit_object[i].setText('500')
                setServoPulse(i, 500, 2000)
        self.resetServos_ = False

    # 选项卡选择标签状态，获取对应舵机数值(Select the label status of the tab to obtain the corresponding servo value)
    def tabindex(self, index):       
        return  [str(self.horizontalSlider_pulse_object[i].value()) for i in self.joints]
    
    def getIndexData(self, index):
        return [str(self.tableWidget.item(index, j).text()) for j in range(2, self.tableWidget.columnCount())]
    
    # 往tableWidget表格添加一行数据的函数(function to add a row of data to the 'tableWidget' table)
    def add_line(self, data):
        self.tableWidget.setItem(data[0], 1, QtWidgets.QTableWidgetItem(str(data[0] + 1)))       
        for i in range(2, len(data) + 1):          
            self.tableWidget.setItem(data[0], i, QtWidgets.QTableWidgetItem(data[i - 1]))

    # 在定位行添加运行图标按钮(add a run icon button in a specific row of the table)
    def icon_position(self):
        toolButton_run = QtWidgets.QToolButton()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(self.path, "resources/index.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        toolButton_run.setIcon(icon)
        toolButton_run.setObjectName("toolButton_run")
        item = self.tableWidget.currentRow()
        self.tableWidget.setCellWidget(item, 0, toolButton_run)
        for i in range(self.tableWidget.rowCount()):
            if i != item:
                self.tableWidget.removeCellWidget(i, 0)
        toolButton_run.clicked.connect(self.action_one)

    def action_one(self):
        self.resetServos_ = True
        item = self.tableWidget.currentRow()
        try:
            timer = int(self.tableWidget.item(self.tableWidget.currentRow(), 2).text())
            for i in range(len(self.joints)):
                servo_id = self.joints[i]
                pulse = int(self.tableWidget.item(item, i + 3).text())
                self.current_positions[servo_id] = pulse
                self.horizontalSlider_pulse_object[servo_id].setValue(pulse)
                self.lineEdit_object[servo_id].setText(str(pulse))
                setServoPulse(servo_id, pulse, timer)   
        except BaseException as e:
            print(e)
            if self.chinese:
                self.message_From('运行出错')
            else:
                self.message_From('Running error')
        self.resetServos_ = False

    # 编辑动作组按钮点击事件(edit action group button click event)
    def button_editaction_clicked(self, name):
        list_data = self.tabindex(self.tableWidget.currentIndex())
        RowCont = self.tableWidget.rowCount()
        item = self.tableWidget.currentRow()
        if name == 'servoPowerDown':
            for j in self.joints:
                unloadServo(j)
            if self.chinese:
                self.message_From('掉电成功')
            else:
                self.message_From('success')
        if name == 'angularReadback':
            self.tableWidget.insertRow(RowCont)    # 增加一行(add a row)
            self.tableWidget.selectRow(RowCont)    # 定位最后一行为选中行(set the last row as the selected row)
            use_time = int(self.lineEdit_time.text())
            data = [RowCont, str(use_time)]
            for j in self.joints:
                pulse = getServoPulse(j)
                if pulse is None:
                    return
                else:
                    data.append(str(pulse))                                       
            if use_time < 20:
                if self.chinese:
                    self.message_From('运行时间必须大于20ms')
                else:
                    self.message_From('Run time must be greater than 20ms')
                return        
            self.add_line(data)
            self.totalTime += use_time
            self.label_TotalTime.setText(str((self.totalTime)/1000.0))            
        if name == 'addAction':    # 添加动作(add action)
            use_time = int(self.lineEdit_time.text())
            data = [RowCont, str(use_time)]
            if use_time < 20:
                if self.chinese:
                    self.message_From('运行时间必须大于20')
                else:
                    self.message_From('Run time must greater than 20')
                return
            self.tableWidget.insertRow(RowCont)    # 增加一行(add a row)
            self.tableWidget.selectRow(RowCont)    # 定位最后一行为选中行(set the last row as the selected row)
            data.extend(list_data)
            self.add_line(data)
            self.totalTime += int(self.lineEdit_time.text())
            self.label_TotalTime.setText(str((self.totalTime)/1000.0))
        if name == 'delectAction':    # 删除动作(delete action)
            if RowCont != 0:
                self.totalTime -= int(self.tableWidget.item(item, 2).text())
                self.tableWidget.removeRow(item)  # 删除选定行(delete the selected row)
                self.label_TotalTime.setText(str((self.totalTime)/1000.0))
        if name == 'delectAllAction':
            result = self.message_delect('此操作会删除列表中的所有动作，是否继续？')
            if result == 0:                              
                for i in range(RowCont):
                    self.tableWidget.removeRow(0)
                self.totalTime = 0
                self.label_TotalTime.setText(str(self.totalTime))
            else:
                pass          
        if name == 'updateAction':    # 更新动作(update action)
            use_time = int(self.lineEdit_time.text())
            data = [item, str(use_time)]            
            if use_time < 20:
                if self.chinese:
                    self.message_From('运行时间必须大于20')
                else:
                    self.message_From('Run time must greater than 20')
                return

            data.extend(list_data)
            self.add_line(data)
            self.totalTime = 0
            for i in range(RowCont):
                self.totalTime += int(self.tableWidget.item(i,2).text())
            self.label_TotalTime.setText(str((self.totalTime)/1000.0))            
        if name == 'insertAction':    # 插入动作(insert action)
            if item == -1:
                return
            use_time = int(self.lineEdit_time.text())
            data = [item, str(use_time)]            
            if use_time < 20:
                if self.chinese:
                    self.message_From('运行时间必须大于20')
                else:
                    self.message_From('Run time must greater than 20')
                return

            self.tableWidget.insertRow(item)       # 插入一行(add a row)
            self.tableWidget.selectRow(item)
            data.extend(list_data)
            self.add_line(data)
            self.totalTime += int(self.lineEdit_time.text())
            self.label_TotalTime.setText(str((self.totalTime)/1000.0))
        if name == 'moveUpAction':
            data_new = [item - 1]
            data = [item]
            if item == 0 or item == -1:
                return
            current_data = self.getIndexData(item)
            uplist_data = self.getIndexData(item - 1)
            data_new.extend(current_data)
            data.extend(uplist_data)
            self.add_line(data_new)           
            self.add_line(data)
            self.tableWidget.selectRow(item - 1) 
        if name == 'moveDownAction':
            data_new = [item + 1]
            data = [item]   
            if item == RowCont - 1:
                return
            current_data = self.getIndexData(item)
            downlist_data = self.getIndexData(item + 1)           
            data_new.extend(current_data)
            data.extend(downlist_data)
            self.add_line(data_new)           
            self.add_line(data) 
            self.tableWidget.selectRow(item + 1)
                             
        for i in range(self.tableWidget.rowCount()):    #刷新编号值(refresh numbering value)
            self.tableWidget.item(i , 2).setFlags(self.tableWidget.item(i , 2).flags() & ~Qt.ItemIsEditable)
            self.tableWidget.setItem(i,1,QtWidgets.QTableWidgetItem(str(i + 1)))
        self.icon_position()

    # 在线运行按钮点击事件(online running button click event)
    def button_run(self, name):
        self.running = True
        if self.tableWidget.rowCount() == 0:
            if self.chinese:
                self.message_From('请先添加动作!')
            else:
                self.message_From('Add action first!')
            self.running = False
        else:
            if name == 'run':
                 if self.Button_Run.text() == '运行' or self.Button_Run.text() == 'Run':
                    if self.chinese:
                        self.Button_Run.setText('停止')
                    else:
                        self.Button_Run.setText('Stop')
                    self.row = self.tableWidget.currentRow()
                    self.tableWidget.selectRow(self.row)
                    self.icon_position()
                    self.timer = QTimer()
                    self.start_run = True
                    self.action_online(self.row)
                    self.use_time_list = [0]
                    if self.checkBox.isChecked():
                        self.timer.timeout.connect(self.operate1)
                        for i in range(self.tableWidget.rowCount() - self.row):
                            use_time = int(self.tableWidget.item(i, 2).text())
                            self.use_time_list.append(use_time)
                        self.use_time_list_ = copy.deepcopy(self.use_time_list)
                        self.timer.start(self.use_time_list_[0])                       
                    else:
                        self.timer.timeout.connect(self.operate2)
                        for i in range(self.tableWidget.rowCount() - self.row):
                            use_time = int(self.tableWidget.item(i, 2).text())
                            self.use_time_list.append(use_time)
                        self.use_time_list_ = copy.deepcopy(self.use_time_list)
                        self.timer.start(self.use_time_list_[0])                         
                 elif self.Button_Run.text() == '停止' or self.Button_Run.text() == 'Stop':
                    self.timer.stop()
                    if self.chinese:
                        self.Button_Run.setText('运行')
                        self.message_From('运行结束!')
                    else:
                        self.Button_Run.setText('Run')
                        self.message_From('Run over!')  
                    self.running = False
            
    def operate1(self):        
        self.use_time_list_.remove(self.use_time_list_[0])
        if self.use_time_list_ != []:
            item = self.tableWidget.currentRow()
            if item != self.tableWidget.rowCount() - 1 and not self.start_run:
                self.tableWidget.selectRow(item + 1)
                self.action_online(item + 1)
                self.icon_position()                           
            else:
                self.start_run = False
            self.timer.start(self.use_time_list_[0])
        else:
            self.tableWidget.selectRow(0)
            self.icon_position()
            self.start_run = True
            self.use_time_list_ = copy.deepcopy(self.use_time_list)
            self.timer.start(self.use_time_list_[0])            
 
    def operate2(self):
        self.use_time_list_.remove(self.use_time_list_[0])
        if self.use_time_list_ != []:
            item = self.tableWidget.currentRow()
            if item != self.tableWidget.rowCount() - 1 and not self.start_run:
                self.tableWidget.selectRow(item + 1)
                self.action_online(item + 1)
                self.icon_position()                           
            else:
                self.start_run = False
            self.timer.start(self.use_time_list_[0])
        else:
            self.timer.stop()
            if self.chinese:
                self.Button_Run.setText('运行')
                self.message_From('运行结束!')
            else:
                self.Button_Run.setText('Run')
                self.message_From('Run over!')            
            self.running = False

    def action_online(self, item):
        try:
            item = self.tableWidget.currentRow()
            timer = int(self.tableWidget.item(self.tableWidget.currentRow(), 2).text())
            for i in range(len(self.joints)):
                servo_id = self.joints[i]
                pulse = int(self.tableWidget.item(item, i + 3).text())
                setServoPulse(servo_id, pulse, timer)
        except BaseException as e:
            print(e)
            self.timer.stop()
            if self.chinese:
                self.Button_Run.setText('运行')
                self.message_From('运行出错!')
            else:
                self.Button_Run.setText('Run')
                self.message_From('Run error!')              

    # 文件打开及保存按钮点击事件(file open and save button click event)
    def button_flie_operate(self, name):
        try:            
            if name == 'openActionGroup':
                dig_o = QFileDialog()
                dig_o.setFileMode(QFileDialog.ExistingFile)
                dig_o.setNameFilter('d6a Flies(*.d6a)')
                openfile = dig_o.getOpenFileName(self, 'OpenFile', self.actdir, 'd6a Flies(*.d6a)')
                # 打开单个文件(open a single file)
                # 参数一：设置父组件；参数二：QFileDialog的标题(Parameter 1: set the parent component; Parameter 2: the title of QFileDialog)
                # 参数三：默认打开的目录，“.”点表示程序运行目录，/表示当前盘符根目录(Parameter 3: the default directory opened, "." represents the program running directory, / represents the root directory of the current drive)
                # 参数四：对话框的文件扩展名过滤器Filter，比如使用 Image files(*.jpg *.gif) 表示只能显示扩展名为.jpg或者.gif文件(Parameter 4: the file extension filter of the dialog box Filter, for example, using Image files (*.jpg *.gif) means that only files with the extension .jpg or .gif can be displayed)
                # 设置多个文件扩展名过滤，使用双引号隔开；“All Files(*);;PDF Files(*.pdf);;Text Files(*.txt)”(set multiple file extension filters, separated by double quotes; "All Files();;PDF Files(.pdf);;Text Files(*.txt)")
                path = openfile[0]
                try:
                    if path != '':
                        rbt = QSqlDatabase.addDatabase("QSQLITE")
                        rbt.setDatabaseName(path)
                        if rbt.open():
                            actgrp = QSqlQuery()
                            if (actgrp.exec("select * from ActionGroup ")):
                                self.tableWidget.setRowCount(0)
                                self.tableWidget.clearContents()
                                self.totalTime = 0
                                while (actgrp.next()):
                                    count = self.tableWidget.rowCount()
                                    self.tableWidget.setRowCount(count + 1)
                                    for i in range(len(self.joints) + 2):
                                        self.tableWidget.setItem(count, i + 1, QtWidgets.QTableWidgetItem(str(actgrp.value(i))))
                                        if i == 1:
                                            self.totalTime += actgrp.value(i)
                                        self.tableWidget.update()
                                        self.tableWidget.selectRow(count)
                                    self.tableWidget.item(count , 2).setFlags(self.tableWidget.item(count , 2).flags() & ~Qt.ItemIsEditable)                                        
                        self.icon_position()
                        rbt.close()
                        self.label_TotalTime.setText(str(self.totalTime/1000.0))
                except Exception as e:
                    print(e)
                    if self.chinese:
                        self.message_From('动作组错误')
                    else:
                        self.message_From('Wrong action format')
                self.action_group_name.setText(str(path)) 
            if name == 'saveActionGroup':
                dig_s = QFileDialog()
                if self.tableWidget.rowCount() == 0:
                    if self.chinese:
                        self.message_From('动作列表是空的哦，没啥要保存的')
                    else:
                        self.message_From('The action list is empty，nothing to save')
                    return
                savefile = dig_s.getSaveFileName(self, 'Savefile', self.actdir, 'd6a Flies(*.d6a)')
                path = savefile[0]
                if os.path.isfile(path):
                    os.system('sudo rm ' + path)
                if path != '':                    
                    if path[-4:] == '.d6a':
                        conn = sqlite3.connect(path)
                    else:
                        conn = sqlite3.connect(path + '.d6a')
                    
                    c = conn.cursor()                    
                    cmd = '''CREATE TABLE ActionGroup([Index] INTEGER PRIMARY KEY AUTOINCREMENT
                    NOT NULL ON CONFLICT FAIL
                    UNIQUE ON CONFLICT ABORT,
                    Time INT,
                    Servo1 INT,
                    Servo2 INT,
                    Servo3 INT,
                    Servo4 INT,
                    Servo5 INT, ''' + ('Servo10 INT);' if DOF == 6 else 'Servo6 INT, Servo10 INT);')
                    c.execute(cmd)
                    if DOF == 6:
                        cmd = "INSERT INTO ActionGroup(Time, Servo1, Servo2, Servo3, Servo4, Servo5, Servo10) VALUES("
                    if DOF == 7:
                        cmd = "INSERT INTO ActionGroup(Time, Servo1, Servo2, Servo3, Servo4, Servo5, Servo6, Servo10) VALUES("
                    for i in range(self.tableWidget.rowCount()):
                        insert_sql = cmd
                        for j in range(2, self.tableWidget.columnCount()):
                            if j == self.tableWidget.columnCount() - 1:
                                insert_sql += str(self.tableWidget.item(i, j).text())
                            else:
                                insert_sql += str(self.tableWidget.item(i, j).text()) + ','
                        
                        insert_sql += ");"
                        c.execute(insert_sql)
                    
                    conn.commit()
                    conn.close()
                    self.button_controlaction_clicked('refresh')
            if name == 'readDeviation':
                for key, value in self.horizontalSlider_dev_object.items():
                    value.setEnabled(True)
                self.readDevOk = True
                dev_data = []
                ids = ''

                for j in self.joints:
                    dev = getServoDeviation(j)
                    if dev is not None:
                        dev_data.append(dev)
                    else:
                        ids += str(" %d"%j)

                if len(dev_data) == len(self.joints):   
                    for i, j in enumerate(self.joints):
                        self.horizontalSlider_dev_object[j].setValue(dev_data[i])
                        self.label_object[j].setText(str(dev_data[i]))
                        
                if ids == '':
                    if self.chinese:
                        self.message_From('读取偏差成功!')
                    else:
                        self.message_From('success!')
                else:
                    if self.chinese:
                        self.message_From(ids + '号舵机偏差读取失败!')
                    else:
                        self.message_From('Failed to read the deviation of' + ids)

            if name == 'downloadDeviation':
                if self.readDevOk:                    
                    for j in self.joints:
                        saveServoDeviation(j)
                        time.sleep(0.05)
                    if self.chinese:
                        self.message_From('下载偏差成功!')
                    else:
                        self.message_From('success!')
                else:
                    if self.chinese:
                        self.message_From('请先读取偏差！')
                    else:
                        self.message_From('Please read the deviation first！')
                self.readDevOK = False
                for k, v in self.horizontalSlider_dev_object.items():
                    v.setEnabled(False)
            if name == 'tandemActionGroup':
                dig_t = QFileDialog()
                dig_t.setFileMode(QFileDialog.ExistingFile)
                dig_t.setNameFilter('d6a Flies(*.d6a)')
                openfile = dig_t.getOpenFileName(self, 'OpenFile', self.actdir, 'd6a Flies(*.d6a)')
                # 打开单个文件(open a single file)
                # 参数一：设置父组件；参数二：QFileDialog的标题(Parameter 1: set the parent component; Parameter 2: the title of QFileDialog)
                # 参数三：默认打开的目录，“.”点表示程序运行目录，/表示当前盘符根目录(Parameter 3: the default directory opened, "." represents the program running directory, / represents the root directory of the current drive)
                # 参数四：对话框的文件扩展名过滤器Filter，比如使用 Image files(*.jpg *.gif) 表示只能显示扩展名为.jpg或者.gif文件(Parameter 4: the file extension filter of the dialog box Filter, for example, using Image files (*.jpg *.gif) means that only files with the extension .jpg or .gif can be displayed)
                # 设置多个文件扩展名过滤，使用双引号隔开；“All Files(*);;PDF Files(*.pdf);;Text Files(*.txt)”(set multiple file extension filters, separated by double quotes; "All Files();;PDF Files(.pdf);;Text Files(*.txt)")
                path = openfile[0]
                try:
                    if path != '':
                        tbt = QSqlDatabase.addDatabase("QSQLITE")
                        tbt.setDatabaseName(path)
                        if tbt.open():
                            actgrp = QSqlQuery()
                            if (actgrp.exec("select * from ActionGroup ")):
                                while (actgrp.next()):
                                    count = self.tableWidget.rowCount()
                                    self.tableWidget.setRowCount(count + 1)
                                    for i in range(len(self.joints) + 2):
                                        if i == 0:
                                            self.tableWidget.setItem(count, i + 1, QtWidgets.QTableWidgetItem(str(count + 1)))
                                        else:                      
                                            self.tableWidget.setItem(count, i + 1, QtWidgets.QTableWidgetItem(str(actgrp.value(i))))
                                        if i == 1:
                                            self.totalTime += actgrp.value(i)
                                        self.tableWidget.update()
                                        self.tableWidget.selectRow(count)
                                    self.tableWidget.item(count , 2).setFlags(self.tableWidget.item(count , 2).flags() & ~Qt.ItemIsEditable)
                        self.icon_position()
                        tbt.close()
                        self.label_TotalTime.setText(str(self.totalTime/1000.0))
                except:
                    if self.chinese:
                        self.message_From('动作组错误')
                    else:
                        self.message_From('Wrong action format')
        except BaseException as e:
            print(e)

    def listActions(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        pathlist = os.listdir(path)
        actList = []
        
        for f in pathlist:
            if f[0] == '.':
                pass
            else:
                if f[-4:] == '.d6a':
                    f.replace('-', '')
                    if f:
                        actList.append(f[0:-4])
                else:
                    pass
        return actList
    
    def refresh_action(self):
        actList = self.listActions(self.actdir)
        actList.sort()
        
        if len(actList) != 0:        
            self.comboBox_action.clear()
            for i in range(0, len(actList)):
                self.comboBox_action.addItem(actList[i])
        else:
            self.comboBox_action.clear()

    # 控制动作组按钮点击事件(control the click event of an action group button)
    def button_controlaction_clicked(self, name):
        if name == 'delectSingle':
            if str(self.comboBox_action.currentText()) != "":
                os.remove(os.path.join(self.actdir, str(self.comboBox_action.currentText()) + ".d6a"))            
                self.refresh_action()
        if name == 'allDelect':
            result = self.message_delect('此操作会删除所有动作组，是否继续？')
            if result == 0:                              
                actList = self.listActions(self.actdir)
                for d in actList:
                    os.remove(os.path.join(self.actdir, d + '.d6a'))
            else:
                pass
            self.refresh_action()
        if name == 'runAction':   # 动作组运行(action group running)
            runActionGroup(self.comboBox_action.currentText())            
        if name == 'stopAction':   # 停止运行(stop running)
            stopActionGroup()
        if name == 'refresh':
            self.refresh_action()
        if name == 'quit':
            sys.exit()

if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
