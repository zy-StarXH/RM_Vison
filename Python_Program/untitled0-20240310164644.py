# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:07:09 2024

@author: 钰钰
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


lower_green = np.array([35, 43, 46])        #为绿色设置阈值用来为之后处理图像准备
upper_green= np.array([77, 255, 255])      #该阈值是在HSV颜色空间下
 
lower_red=np.array([0,43,46])
upper_red= np.array([10, 255, 255]) 

lower_blue=np.array([100,50,100])
upper_blue= np.array([124, 255, 255]) 

 


#颜色提取
def img_process(img,lower,upper):
    """返回处理后只留下指定颜色的图像（其余为黑色）
        img：原图像；lower：最低阈值；upper：最高阈值"""
   
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          #将BGR图像转化为HSV图像
    aussian=cv2.GaussianBlur(hsv,(5,5),1)              #进行高斯滤波操作，去除噪声
    mask = cv2.inRange(aussian, lower, upper)              #二值化处理
    res = cv2.bitwise_and(img, img, mask = mask)        #进行位与运算
    return res                                          

#绿灯轮廓描绘
def cnts_draw(img,res):
    """ img：原图像；res：上函数处理后图像色"""
    canny = cv2.Canny(res,100,200)#Canny边缘检测算法
    contours, hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#寻找图像轮廓的函数
    if len(contours) == 0:#传递到max函数中的轮廓不能为空
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == 27: #等待按键ESC按下
            return
    else:
        #draw_img=img.copy()
        max_cnt = max(contours , key = cv2.contourArea)#找到轮廓中最大的一个
       #绘制近似圆
        (x,y),radius=cv2.minEnclosingCircle(max_cnt)
        center=(int(x),int(y))
        radius=int(radius)
        img=cv2.circle(img,center,radius,(0,255,0),2)
        cv2.imshow('video', img)#展示原图
        if cv2.waitKey(1) & 0xFF == 27: #等待按键ESC按下
            return

#识别长灯条
def idefi_strip(img,lower,upper):
    #图像处理
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          
    aussian=cv2.GaussianBlur(hsv,(5,5),1)              
    #进行高斯滤波操作，去除噪声
    mask = cv2.inRange(aussian, lower, upper)              
    res = cv2.bitwise_and(img, img, mask = mask)        
    canny = cv2.Canny(mask,100,200)
    contours, hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:#传递到max函数中的轮廓不能为空
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == 27: #等待按键ESC按下
            return
    else:
        #绘制近似矩阵
        max_cnt = max(contours , key = cv2.contourArea)
        (x,y,w,h) = cv2.boundingRect(max_cnt)
        if(((h/w)>22.4)&((h/w)<22.6)):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            cv2.imshow('video', img)
            if cv2.waitKey(1) & 0xFF == 27: #等待按键ESC按下
                return
    
#打开摄像头，设置相机参数
image = cv2.VideoCapture('qianshaozhan1.mp4')
#图像宽度
'''image.set(3,600)
image.set(cv2.CAP_PROP_FRAME_WIDTH,600)
#图像高度
image.set(4,500)
image.set(cv2.CAP_PROP_FRAME_HEIGHT,500)
#视频帧率
image.set(5, 30)  #设置帧率
image.set(cv2.CAP_PROP_FPS, 30)
#解码方式四字符
image.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
#图像亮度
image.set(cv2.CAP_PROP_BRIGHTNESS, 63)
#图像对比度
image.set(cv2.CAP_PROP_CONTRAST, 0)   '''
#图像曝光度
image.set(cv2.CAP_PROP_EXPOSURE, 2000) 


if not image.isOpened():
    print("Cannot open camera")
    exit() 
    
while (image.isOpened()):
    #逐帧捕获
    ret, frame = image.read()
    if frame is None:
        break;
    if ret==True:
        res=img_process(frame,lower_green,upper_green)
        cnts_draw(frame,res)
        idefi_strip(frame,lower_red,upper_red)#该处设置为红，可根据具体情况进行调整
        if cv2.waitKey(1) & 0xFF == 27: #等待按键ESC按下
            break

image.release()
cv2.destroyAllWindows()
       
       