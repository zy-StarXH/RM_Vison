import cv2
import numpy as np


# 读取图像
image = cv2.imread('Duihuanzhan.png')
# 读取图像用于画框
RECimg = cv2.imread('Duihuanzhan.png')

# 灰度化图像，
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 转换为HSV色彩空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色和蓝色范围：
lower_red = np.array([156, 50, 50])
upper_red = np.array([180, 255, 255])

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([124, 255, 255])

red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

red_img = cv2.bitwise_and(image, image, mask=red_mask)
blue_img = cv2.bitwise_and(image, image, mask=blue_mask)
# 检测兑换站的颜色：
if cv2.countNonZero(red_mask) > cv2.countNonZero(blue_mask):
    lower = lower_red
    upper = upper_red
else:
    lower = lower_blue
    upper = upper_blue


# 创建掩码，将符合条件的像素设置为255白色，不符合条件的像素设置为0黑色
mask = cv2.inRange(hsv_image, lower, upper)

# 应用掩码到原始图像上，只显示符合条件的区域
result = cv2.bitwise_and(image, image, mask=mask)
# 再次对颜色分割的图像灰度化
gray2 = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)

# 对掩码图像进行二值化处理，然后寻找轮廓
s, binary2 = cv2.threshold(gray2,50,255,cv2.THRESH_BINARY)

# 寻找到的轮廓可以使用于draw contour和rectangle
contours, _ = cv2.findContours(binary2,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)


# 创建包围盒
bounding_box = None
# contour在图像中遍历轮廓并对x y w h赋值
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    if bounding_box is None:
        bounding_box = (x, y, x + w, y + h)
    else:# 寻找最大的包围盒
        bounding_box = (min(bounding_box[0],x),min(bounding_box[1],y),max(bounding_box[2],x + w),max(bounding_box[3],y +h))
# 绘制方框
cv2.rectangle(RECimg,bounding_box[0:2],bounding_box[2:4],(0,0,255),2)
# 绘制轮廓
cv2.drawContours(image,contours,-1,(0,0,255),5)

# 显示结果图像
cv2.imshow('Mask result', result)
cv2.imshow('contours',image)
cv2.imshow('Mask binary',binary2)
cv2.imshow('REC img',RECimg)
cv2.waitKey(0)
cv2.destroyAllWindows()