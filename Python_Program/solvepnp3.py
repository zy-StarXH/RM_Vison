import cv2
import numpy as np

posture = 'normal'
ip_address = '10.13.81.95'  # 使用droidcam调用手机摄像头，应自行更改ip地址
port = 4747  # 端口
# capture = cv2.VideoCapture(f'http://{ip_address}:{port}/mjpegfeed')  # 使用网站调用摄像头
capture = cv2.VideoCapture('target_video4.mp4')
# 定义黄色范围：
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])
# 假设的3D世界坐标点（单位：米）
objectPoints = np.array([[-100, -75, 175],
                         [-100, 75, 175],
                         [-100, 75, 25],
                         [-100, -75, 25]], dtype=np.float32)

fx, fy, cx, cy = 583.915251506799, 583.762620412079, 313.835192027846, 227.114376996309  # 示例值
k1, k2, p1, p2, k3 = 0.0243295245551931, 0.0170719495074567, 0, 0, 0

def BubbleSort_X(arr):  # 升序排列
    for i in range(1, len(arr)):
        for j in range(0, len(arr)-i):
            if arr[j][0] > arr[j+1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def BubbleSort_Y(arr):  # 升序排列
    for i in range(1, len(arr)):
        for j in range(0, len(arr)-i):
            if arr[j][1] > arr[j+1][1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def Showimg():
    # 叠加图像，显示轮廓：
    cv2.imshow('yellow_img', yellow_img)
    cv2.imshow('result', result_image)
    cv2.imshow('canny', r1)


# 检测模式选择：
while True:
    ret, image = capture.read()
    # 提取画面中的黄色部分，找到金矿石的大致位置：
    resized_img = image
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)  # 创建掩模
    yellow_img = cv2.bitwise_and(resized_img, resized_img, mask=mask)
    yellow_img[np.where(mask == 0)] = [0, 0, 0]  # 将非黄色区域设置为黑色
    gray = cv2.cvtColor(yellow_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray, (5, 5), 0)
    s, binary = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)
    r1 = cv2.Canny(binary, 100, 255)
    contours, _ = cv2.findContours(r1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算每个轮廓的面积
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    highlighted_image = np.zeros_like(resized_img)
    bounding_boxes = []
    left_up_pos = ()
    left_down_pos = ()
    right_up_pos = ()
    right_down_pos = ()

    # 遍历轮廓，寻找合适的标记
    for contour in contours:
        area = cv2.contourArea(contour)
        # 面积和长宽比判断
        if 100 <= area <= 8000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            epsilon = 0.03 * cv2.arcLength(contour, True)  # 计算轮廓的周长，然后乘以0.02作为精度参数
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 判断多边形形状（假设我们只对四边形或六边形感兴趣）
            if len(approx) == 4 and 0.6 <= aspect_ratio <= 1.4:
                # 将边界框添加到列表中
                bounding_boxes.append((x, y, w, h))
                cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
            if len(approx) == 6 and 0.6 <= aspect_ratio <= 1.4:
                bounding_boxes.append((x, y, w, h))
                cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
    if len(bounding_boxes) >= 4:
        corrected_X = BubbleSort_X(bounding_boxes)
        corrected_Y_1 = BubbleSort_Y(corrected_X[0:2])  # 左上角和左下角
        corrected_Y_2 = BubbleSort_Y(corrected_X[2:4])  # 右上角和右下角
        if corrected_Y_2:
            right_up_pos = (corrected_Y_2[0][0]+corrected_Y_2[0][2], corrected_Y_2[0][1])
            if len(corrected_Y_2) == 2:
                right_down_pos = (corrected_Y_2[1][0]+corrected_Y_2[1][2], corrected_Y_2[1][1]+corrected_Y_2[1][3])
        if len(corrected_Y_1) == 2:
            left_down_pos = (corrected_Y_1[1][0], corrected_Y_1[1][1]+corrected_Y_1[1][3])
        if corrected_Y_1:
            left_up_pos = (corrected_Y_1[0][0], corrected_Y_1[0][1])

    result_image = cv2.addWeighted(resized_img, 0.4, highlighted_image, 0.6, 0)
    Showimg()
    # 假设的2D图像坐标点（单位：像素）
    # 这些点通常是通过某种特征检测算法（如SIFT, SURF, ORB等）和匹配算法得到的
    if len(left_up_pos) and len(left_down_pos) and len(right_down_pos) and len(right_up_pos):
        imagePoints = np.array([[left_up_pos[0], left_up_pos[1]],
                                [right_up_pos[0], right_up_pos[1]],
                                [right_down_pos[0], right_down_pos[1]],
                                [left_down_pos[0], left_down_pos[1]]], dtype=np.float32)

        # 相机内参矩阵，这通常是通过相机标定得到的
        # 这里我们使用一些假设的值作为示例
        cameraMatrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]], dtype=np.float32)

        distCoeffs = np.array([k1, k2, p1, p2, k3])

        # 使用solvePnP求解相机姿态
        _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

        # rvec和tvec分别是旋转向量和平移向量
        # 你可以使用cv2.Rodrigues将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        length = 100
        center = (resized_img.shape[1] // 2, resized_img.shape[0] // 2)
        cv2.line(resized_img, left_up_pos, right_down_pos, (255, 255, 255), 3)
        cv2.line(resized_img, right_up_pos, left_down_pos, (255, 255, 255), 3)
        cv2.circle(resized_img, left_up_pos, 2, (0,255,255), 15)
        cv2.circle(resized_img, right_down_pos, 2, (255, 0, 255), 15)
        X_axis = ((cameraMatrix @ (R @ np.array([1, 0, 0]).T)) / R[2, 0])
        
        # X_axis_show = (int(X_axis[0] + resized_img.shape[1] // 2), int(X_axis[1] + resized_img.shape[0] // 2))
        # X_axis = (int(fx * R[0, 0] / R[2, 0] + cx), int(fy * R[1, 0] / R[2, 0] + cy))
        # Y_axis = (int(fx * R[0, 1] / R[2, 1] + cx), int(fy * R[1, 1] / R[2, 1] + cy))
        # Z_axis = (int(fx * R[0, 2] / R[2, 2] + cx), int(fy * R[1, 2] / R[2, 2] + cy))
        # 绘制相机姿态
        axis_length = 100
        axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1,
                                                                                                                     3)
        projected_axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, cameraMatrix, distCoeffs)
        projected_axis_points = np.int32(projected_axis_points)
        print(projected_axis_points[0][0])
        for i in range(1, len(projected_axis_points)):
            cv2.line(resized_img, tuple(projected_axis_points[0].ravel()), tuple(projected_axis_points[i].ravel()),
                     (255, 0, 0), 2)

        # cv2.line(resized_img, center, X_axis, (255, 0, 0), 5)
        # cv2.line(resized_img, center, Y_axis, (0, 255, 0), 5)
        # cv2.line(resized_img, center, Z_axis, (0, 0, 255), 5)
        cv2.imshow('final img', resized_img)
    key = cv2.waitKey(10)  # 停顿10ms后进入下一次循环
    if key == 32:  # 如果检测到按下空格键
        break

