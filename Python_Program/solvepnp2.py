import numpy as np
import cv2

left_arrow_sign = 0
left_square_sign = 0
right_arrow_sign = 0
right_square_sign = 0
rd_arrow_sign = 0
rd_square_sign = 0
ld_arrow_sign = 0
ld_square_sign = 0
posture = 'normal'
# 定义黄色范围：
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
silver_lower = np.array([0, 0, 50])
silver_upper = np.array([180, 255, 255])
capture = cv2.VideoCapture(1)
# 提取画面中的黄色部分，找到金矿石的大致位置：
while True:
    ret, image = capture.read()
    resized_img = cv2.resize(image, (500, 500))
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)  # 创建掩模
    yellow_img = cv2.bitwise_and(resized_img, resized_img, mask=mask)
    yellow_img[np.where(mask == 0)] = [0, 0, 0]  # 将非黄色区域设置为黑色
    gray = cv2.cvtColor(yellow_img, cv2.COLOR_BGR2GRAY)
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
        area = w1*h1
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
        cv2.imshow('ROI', ROI)
        gray2 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        _, binary2 = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)

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
                if x <= resized_roi.shape[1] // 2 and y <= resized_roi.shape[0] // 2:
                    # 计算轮廓的近似多边形
                    epsilon = 0.02 * cv2.arcLength(contour, True)  # 计算轮廓的周长，然后乘以0.02作为精度参数
                    approx = cv2.approxPolyDP(contour, epsilon, True)  # 计算轮廓的近似多边形，True”表示多边形是闭合的
                    # 获取包含轮廓的最小矩形的长宽：
                    x2, y2, w2, h2 = cv2.boundingRect(contour)
                    u1, v1 = x2, y2

                # 检测右上角四分之一的区域：具体检测方式与上方相同
                if x >= resized_roi.shape[1] // 2 and y <= resized_roi.shape[0] // 2:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    x2, y2, w2, h2 = cv2.boundingRect(contour)
                    u2, v2 = x2 + w2, y2

                # 检测左下角标记：
                if x <= resized_roi.shape[1] // 2 and y >= resized_roi.shape[0] // 2:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    x2, y2, w2, h2 = cv2.boundingRect(contour)
                    u3, v3 = x2, y2 + h2

                # 检测右下角：
                if x >= resized_roi.shape[1] // 2 and y >= resized_roi.shape[0] // 2:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    x2, y2, w2, h2 = cv2.boundingRect(contour)
                    u4, v4 = x2 + w2, y2 + h2

    # 假设的3D世界坐标点（单位：米）
    objectPoints = np.array([[100, -75, 175],
                             [100, 75, 175],
                             [100, 75, 25],
                             [100, -75, 25]], dtype=np.float32)

    fx, fy, cx, cy = 583.915251506799, 583.762620412079, 313.835192027846, 227.114376996309  # 示例值
    k1, k2, p1, p2, k3 = 0.0243295245551931, 0.0170719495074567, 0, 0, 0

    # 假设的2D图像坐标点（单位：像素）
    # 这些点通常是通过某种特征检测算法（如SIFT, SURF, ORB等）和匹配算法得到的
    try:
        imagePoints = np.array([[u1, v1],
                                [u2, v2],
                                [u4, v4],
                                [u3, v3]], dtype=np.float32)
    except:
        cv2.imshow('1', resized_img)
        key = cv2.waitKey(10)
        if key == 32:  # 如果检测到按下空格键
            break
        continue
    else:
        # 相机内参矩阵，这通常是通过相机标定得到的
        # 这里我们使用一些假设的值作为示例
        cameraMatrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]], dtype=np.float32)

        # 畸变系数，这里我们假设没有畸变，所以所有值都是0
        distCoeffs = np.array([k1, k2, p1, p2, k3])

        # 使用solvePnP求解相机姿态
        _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

        # rvec和tvec分别是旋转向量和平移向量
        # 你可以使用cv2.Rodrigues将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        center = (resized_roi.shape[1] // 2, resized_roi.shape[0] // 2)

        # 定义向量长度（以像素为单位）
        length = 100

        # 绘制X轴（红色）
        cv2.line(resized_img, center, (int(center[0] + length * R[0, 0]), int(center[1] - length * R[0, 1])),
                 (0, 0, 255), 2)

        # 绘制Y轴（绿色）
        cv2.line(resized_img, center, (int(center[0] - length * R[1, 0]), int(center[1] + length * R[1, 1])),
                 (0, 255, 0), 2)

        # 绘制Z轴（蓝色）
        cv2.line(resized_img, center, (int(center[0] - length * R[2, 0]), int(center[1] - length * R[2, 1])),
                 (255, 0, 0), 2)
        cv2.imshow('Camera Axes', resized_img)
        print("Rotation Matrix:\n", R)
        key = cv2.waitKey(10)  # 停顿10ms后进入下一次循环
        if key == 32:  # 如果检测到按下空格键
            break
    # 注意: u1, v1, u2, v2, u3, v3, u4, v4, fx, fy, cx, cy 是你需要替换的具体值