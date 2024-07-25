import cv2
import numpy as np

# 加载图像
image = cv2.imread('jinkuangshi.jpg')

# 将图像从BGR转换为HSV色彩空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义蓝色在HSV色彩空间中的范围
lower_blue = np.array([50, 155, 155])
upper_blue = np.array([130, 255, 255])

# 分割颜色，检测图像中的蓝色并设置为255白色 不符合的像素设置为0黑色
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 对原始图像和掩码进行位运算
res = cv2.bitwise_and(image, image, mask=mask)

# 找到掩码中的轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算四个灯条的包围盒（bounding box）
bounding_box = None
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if bounding_box is None:
        bounding_box = (x, y, x + w, y + h)  # 初始化包围盒
    else:
        bounding_box = (min(bounding_box[0], x), min(bounding_box[1], y), max(bounding_box[2], x + w),
                        max(bounding_box[3], y + h))  # 更新包围盒

# 在原始图像上绘制矩形框（使用包围盒）
cv2.rectangle(image, bounding_box[0:2], bounding_box[2:4], (0, 255, 0), 2)

# 显示结果图像
cv2.imshow('Detected Blue Light Bars', image)
cv2.waitKey(0)
cv2.destroyAllWindows()