#!/usr/bin/env python3
# encoding: utf-8
import os
import time
import threading
import sqlite3 as sql
import servo_controller as controller

runningAction = False
stopRunning = False
online_action_num = None
online_action_times = -1
update_ok = False
action_group_finish = True
setServoPulse = None

action_path = os.path.split(os.path.realpath(__file__))[0]

def stop_servo():
    for i in range(16):
        stopBusServo(i+1) 

def stop_action_group():
    global stopRunning, online_action_num, online_action_times, update_ok
    update_ok = False
    stopRunning = True
    online_action_num = None
    online_action_times = -1
    time.sleep(0.1)

def action_finish():
    global action_group_finish
    return action_group_finish  

def runAction(actNum):
    '''
    运行动作组，无法发送stop停止信号(Running action group, unable to send a 'stop' signal)
    :param actNum: 动作组名字 ， 字符串类型(name of the action group, string type)
    :param times:  运行次数(running times)
    :return:
    '''
    global runningAction
    global stopRunning
    global online_action_times
    if actNum is None:
        return
    actNum = os.path.join(action_path, 'ActionGroups', actNum + ".d6a")
    stopRunning = False
    if os.path.exists(actNum) is True:
        if runningAction is False:
            runningAction = True
            ag = sql.connect(actNum)
            cu = ag.cursor()
            cu.execute("select * from ActionGroup")
            while True:
                act = cu.fetchone()
                if stopRunning is True:
                    stopRunning = False                   
                    break
                if act is not None:
                    for i in range(0, len(act)-2, 1):
                        if i < (len(act) - 3):
                            controller.setServoPulse(i+1, act[2 + i], act[1])
                        else:
                            controller.setServoPulse(10, act[2 + i], act[1])
                    time.sleep(float(act[1])/1000.0)
                else:   # 运行完才退出(exit only after finishing the execution)
                    break
            runningAction = False
            
            cu.close()
            ag.close()
    else:
        runningAction = False
        print("未能找到动作组文件")

def online_thread_run_acting():
    global online_action_times, online_action_num, update_ok, action_group_finish
    while True:
        if update_ok:
            if online_action_times == 0:
                # 无限次运行(run infinitely)
                if action_group_finish:
                    action_group_finish = False
                runAction(online_action_num)                
            elif online_action_times > 0:
                # 有次数运行(run for a certain number of times)
                if action_group_finish:
                    action_group_finish = False
                runAction(online_action_num)
                online_action_times -= 1    # 运行完成后，进入空载(After finishing the execution, enter the no-load state)
                if online_action_times == 0:
                    online_action_times = -1
            else:
                # 空载(no-load)
                if not action_group_finish:
                    action_group_finish = True
                time.sleep(0.001)
        else:
            if not action_group_finish:
                action_group_finish = True
            time.sleep(0.001)
            
def start_action_thread():
    th1 = threading.Thread(target=online_thread_run_acting)
    th1.setDaemon(True)  # 设置为后台线程，这里默认是True(Set as a background thread, which is 'True' by default)
    th1.start()
    
def change_action_value(actNum, actTimes):
    global online_action_times, online_action_num, update_ok, stopRunning, action_group_finish
    
    if action_group_finish:
        online_action_times = actTimes
        online_action_num = actNum
        stopRunning = False
        update_ok = True
