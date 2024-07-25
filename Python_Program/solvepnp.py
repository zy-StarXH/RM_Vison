import cv2
import numpy as np


# 初始化鼠标回调函数的全局变量
click_event = None
click_point = [None, None, None, None]


# 鼠标回调函数
def mouse_event(event, x, y, flags, param):
    global click_event
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        click_event = (x, y)
        a1 = (x, y)
        click_point.append(a1)
        print(f"Clicked at ({x}, {y})")


img = cv2.imread('jinkuangshi.jpg')

# 创建一个窗口来显示图像
cv2.namedWindow('image')
# 设置鼠标回调函数
cv2.setMouseCallback('image', mouse_event)

# 显示图像
while True:
    resized_img = cv2.resize(img, (500, 500))
    cv2.imshow('image', resized_img)

    # 按'q'键退出
    if click_point[3] is not None:
        imagePoints = np.array([
            [click_point[0]],  # 假设这是左上角的点
            [click_point[1]],  # 右上角的点
            [click_point[2]],  # 右下角的点
            [click_point[3]]  # 左下角的点
        ], dtype=np.float32)
        break

    # 销毁所有窗口
cv2.destroyAllWindows()
# 初始化 objectPoints 和 imagePoints 的数据
# 注意：在实际应用中，你需要填充这些点的数据
obPoints = np.array([
    [0, 0],  # 假设这是左上角的点
    [29.5, 0],  # 右上角的点
    [0, 629.5],  # 右下角的点
    [29.5, 29]  # 左下角的点
], dtype=np.float32)

# 假设我们有一些内参和畸变系数
# 这些值通常需要通过相机标定得到
fx, fy, cx, cy = 592.59016206, 590.84858199, 310.09956679, 230.53149091  # 示例值
k1, k2, p1, p2, k3 = 0.01225467, 0.26837427, 0.0034651, 0.00189021, -0.77383142  # 示例畸变系数

# 创建相机内参数矩阵
cameraMatrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])

# 创建相机畸变系数矩阵
distCoeffs = np.array([k1, k2, p1, p2, k3])

# solvePnP 需要至少4个匹配点对
# 注意：在实际应用中，你需要确保 objectPoints 和 imagePoints 的长度（个数）相等且大于或等于4

# 假设 objectPoints 和 imagePoints 已经被填充并且长度相等
# 这里仅作为示例，不实际填充数据


# 调用 solvePnP 函数
rvec, tvec, _ = cv2.solvePnP(obPoints, imagePoints, cameraMatrix, distCoeffs)

# 检查是否成功
if rvec is not None and tvec is not None:
    # 获取旋转矩阵
    rotationMatrix, _ = cv2.Rodrigues(rvec)

    # 输出结果
    print("Rotation Vector:")
    print(rvec)
    print("Translation Vector:")
    print(tvec)
    print("Rotation Matrix:")
    print(rotationMatrix)

    # 注意：在Python中，通常不需要显式地返回0，因为Python脚本执行完毕后会自然退出