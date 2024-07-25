import cv2
import numpy as np

# 加载要检测的图像特征
target_image = cv2.imread('target_video9.mp4', 0)  # 确保路径正确，0表示以灰度模式加载
orb = cv2.ORB_create()

# 计算目标图像的关键点和描述符
target_keypoints, target_descriptor = orb.detectAndCompute(target_image, None)

# 创建BFMatcher对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 初始化摄像头
cap = cv2.VideoCapture(0)

while True:
    # 从摄像头读取画面
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算摄像头画面的关键点和描述符
    frame_keypoints, frame_descriptor = orb.detectAndCompute(gray_frame, None)

    # 使用BFMatcher进行匹配
    matches = bf.match(target_descriptor, frame_descriptor)

    # 根据匹配结果，提取强匹配点
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:15]  # 假设前15个是好的匹配点

    # 画出匹配点
    frame_matches = cv2.drawMatches(target_image, target_keypoints, gray_frame, frame_keypoints, good_matches, None, flags=2)

    # 显示匹配结果
    cv2.imshow("Matches", frame_matches)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
