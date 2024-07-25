import cv2
import numpy as np

# 加载图像
image = cv2.imread('jinkuangshi.jpg', 0)  # 0 表示以灰度模式读取图像
resized_img = cv2.resize(image, (500, 500))
# 设置 Shi-Tomasi 角落检测的参数
maxCorners = 100  # 最多检测到的角落数
qualityLevel = 0.3  # 角落检测的最小质量水平
minDistance = 7  # 角落之间的最小欧几里得距离

# 使用 goodFeaturesToTrack 函数检测角落
corners = cv2.goodFeaturesToTrack(resized_img, maxCorners, qualityLevel, minDistance)

# corners 是一个 N x 1 x 2 的 numpy 数组，其中 N 是检测到的角落数
# 每个角落由 (x, y) 坐标表示

# 绘制检测到的角落
for corner in corners:
    x, y = map(int, corner.ravel())  # 使用 map 函数将浮点数转换为整数
    cv2.circle(resized_img, (x, y), 3, (0, 255, 0), -1)  # 绘制圆，使用 BGR 颜色（白色）

# 显示图像
cv2.imshow('Corners', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
