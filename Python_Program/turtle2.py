import cv2
import numpy as np
import pyautogui
import time

# 设置捕获屏幕的分辨率
screen_size = (2560, 1600)  # 根据你的屏幕分辨率进行调整

while True:
    # 使用 pyautogui 捕获屏幕截图
    screenshot = pyautogui.screenshot(region=screen_size)

    # 将 PIL Image 转换为 numpy array，因为 OpenCV 使用 numpy array 来处理图像
    frame = np.array(screenshot)

    # OpenCV 使用 BGR 顺序，而 pyautogui 捕获的图像是 RGB 顺序，因此需要转换
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 在 OpenCV 窗口中显示捕获的帧
    cv2.imshow('Desktop Capture', frame)

    # 等待 1 毫秒
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

    # 释放资源并关闭窗口
cv2.destroyAllWindows()
