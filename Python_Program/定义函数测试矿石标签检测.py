import cv2
import numpy as np

capture = cv2.VideoCapture('target_video4.mp4')
# 定义黄色范围：
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])


def find_sign(image):
    # 提取画面中的黄色部分，找到金矿石的大致位置：
    resized_img = cv2.resize(image, (500, 500))
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)  # 创建掩模
    yellow_img = cv2.bitwise_and(resized_img, resized_img, mask=mask)
    yellow_img[np.where(mask == 0)] = [0, 0, 0]  # 将非黄色区域设置为黑色
    # 高斯模糊降低噪声，寻找轮廓：
    blurred = cv2.GaussianBlur(yellow_img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    s, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 当目标不在视野内时，保活程序
    if len(contours) == 0:
        black_img = np.zeros((500, 500, 3), np.uint8)
        resized_img = black_img
        r1 = black_img
        resized_roi = black_img
        result_image = black_img
    else:
        max_contour = max(contours, key=cv2.contourArea)
        # 计算最小外接矩形,判断画面中是否有金矿石
        rect = cv2.minAreaRect(max_contour)
        w1, h1 = rect[1]
        area = w1 * h1
        if area < 1000:
            print("未检测到金矿石！")
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 获取ROI
        x, y, w, h = cv2.boundingRect(box)
        rotated_roi = resized_img[y:y + h, x:x + w]

        # 调整ROI的方向
        if rotated_roi.size != 0:
            # 放大ROI
            scale_factor = 2
            scaled_roi = cv2.resize(rotated_roi, None, fx=scale_factor, fy=scale_factor,
                                    interpolation=cv2.INTER_LINEAR)
            resized_roi = cv2.resize(scaled_roi, (500, 500))
        else:
            width, height = 500, 500
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            blank[1, 1] = (255, 255, 255)
            resized_roi = blank

        # 找到左上角标记，并使用形状等计算几何判断
        # 对ROI区进行轮廓处理，形状判断
        ROI = resized_roi
        gray2 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        _, binary2 = cv2.threshold(gray2, 100, 255, cv2.THRESH_BINARY)

        r1 = cv2.Canny(binary2, 100, 255)
        contours2, _ = cv2.findContours(r1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化一个和原图像大小相同的空白图像，用于绘制符合条件的轮廓
        highlighted_image = np.zeros_like(resized_roi)

        # 遍历所有轮廓
        for contour in contours2:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)

            # 检查面积是否在指定范围内,这个面积范围包括了箭头形标记以及方块标记
            if 1000 <= area <= 8000:
                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                # 检查边界框是否位于图像的左上角区域
                # 这里假设左上角区域是图像左上角的1/4部分
                if x <= resized_roi.shape[1] // 4 and y <= resized_roi.shape[0] // 4:
                    # 计算轮廓的近似多边形
                    epsilon = 0.02 * cv2.arcLength(contour, True)  # 计算轮廓的周长，然后乘以0.02作为精度参数
                    approx = cv2.approxPolyDP(contour, epsilon, True)  # 计算轮廓的近似多边形，True”表示多边形是闭合的
                    # 获取包含轮廓的最小矩形的长宽：
                    x2, y2, w2, h2 = cv2.boundingRect(contour)
                    aspect_ratio = w2 / float(h2)
                    # 判断是否为六边形：若是，则初步筛选为箭头型标记
                    if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                        # 绘制轮廓
                        cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                        print('left_arrow_sign get')
                        left_arrow_sign = 1
                        # 在空白图像上绘制轮廓
                        cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                    if len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:
                        # 绘制轮廓
                        cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                        print('left_square_sign get')
                        left_square_sign = 1
                        # 在空白图像上绘制轮廓
                        cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                # 检测右上角四分之一的区域：具体检测方式与上方相同
                if x >= resized_roi.shape[1] // 4 and y <= resized_roi.shape[0] // 4:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    x2, y2, w2, h2 = cv2.boundingRect(contour)
                    aspect_ratio = w2 / float(h2)
                    if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                        cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                        print('right_arrow_sign get')
                        right_arrow_sign = 1
                        cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                    if len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:
                        cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                        print('right_square_sign get')
                        right_square_sign = 1
                        cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
        # 将绘制了轮廓的图像与原图像合并，以突出显示轮廓
        result_image = cv2.addWeighted(resized_roi, 0.4, highlighted_image, 0.6, 0)
    return result_image


while capture.isOpened():
    retval, video = capture.read()
    process = find_sign(video)
    cv2.imshow('process', process)
    key = cv2.waitKey(10)  # 停顿1ms后进入下一次循环
    if key == 32:  # 如果检测到按下空格键
        break
cv2.destroyAllWindows()
capture.release()
