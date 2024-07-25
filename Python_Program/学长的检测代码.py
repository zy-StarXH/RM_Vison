import cv2
import numpy as np

# 读取图像
image = cv2.imread("test.png")

# 将图像从BGR颜色空间转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#以下是蓝色的
lower_blue = np.array([10, 43, 46])
upper_blue = np.array([124, 255, 255])
#以下是红色的
lower_red1 = np.array([0, 43, 46])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([156, 43, 46])
upper_red2 = np.array([180, 255, 255])
#以下是绿色的
lower_green = np.array([78,43,46])
upper_green = np.array([77,255,255])

# 创建掩模，通过颜色范围进行颜色分割
mask_blue= cv2.inRange(hsv_image, lower_blue, upper_blue)
mask_red1= cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2= cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2
mask_green= cv2.inRange(hsv_image, lower_green, upper_green)

# 在原始图像上应用掩码
result_blue = cv2.bitwise_and(image, image, mask_blue)
result_red = cv2.bitwise_and(image, image, mask_red)
result_green = cv2.bitwise_and(image, image, mask_green)

# 查找轮廓
contours1, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours3, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历找到的轮廓
for contour in contours1:
    # 计算轮廓的中心和半径
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    # 在原始图像上绘制圆形和中心
    cv2.circle(image, center, radius, (0, 255, 0), 2)
    cv2.circle(image, center, 3, (0, 255, 0), -1)

for contour in contours2:
    # 计算轮廓的中心和半径
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    # 在原始图像上绘制圆形和中心
    cv2.circle(image, center, radius, (0, 255, 0), 2)
    cv2.circle(image, center, 3, (0, 255, 0), -1)

for contour in contours3:
    # 计算轮廓的中心和半径
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    # 在原始图像上绘制圆形和中心
    cv2.circle(image, center, radius, (0, 255, 0), 2)
    cv2.circle(image, center, 3, (0, 255, 0), -1)

# 显示结果图像
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
