import cv2
import numpy as np
import math

# 内参数矩阵
Camera_intrinsic_8mm = {
    "mtx": np.array([[592.59016206, 0, 310.09956679],
                      [0, 590.84858199, 230.53149091],
                      [0, 0, 1]], dtype=np.double),
    "dist": np.array([[0.01225467, 0.26837427, 0.0034651, 0.00189021,
                       -0.77383142]], dtype=np.double),
}

Width_set = 1280  # 设置分辨率宽
Height_set = 960  # 设置分辨率高
framerate_set = 50  # 设置帧率
num = 100000  # 采集帧率次数
id = 0

device_manager = gx.DeviceManager()  # 创建设备对象
dev_num, dev_info_list = device_manager.update_device_list()  # 枚举设备，即枚举所有可用的设备
if dev_num == 0:
    print("Number of enumerated devices is 0")
else:
    print("")
    print("**********************************************************")
    print("创建设备成功，设备号为:%d" % dev_num)
cam = device_manager.open_device_by_sn(dev_info_list[id].get("sn"))
camera_id = dev_info_list[id].get("sn")
print("")
print("**********************************************************")
print("打开彩色摄像机成功，SN号为：%s" % camera_id)

cam.Width.set(Width_set)
cam.Height.set(Height_set)
# 设置连续采集
cam.TriggerMode.set(gx.GxSwitchEntry.OFF)  # 设置触发模式
# cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
# 设置帧率
cam.AcquisitionFrameRate.set(framerate_set)
print("")
print("**********************************************************")
print("用户设置的帧率为:%d fps" % framerate_set)
framerate_get = cam.CurrentAcquisitionFrameRate.get()  # 获取当前采集的帧率
print("当前采集的帧率为:%d fps" % framerate_get)

cam.stream_on()

#这里是一下估测的核心代码
while True:
    # 采集图像
    raw_image = cam.data_stream[0].get_image()  # 打开第0通道数据流
    if raw_image is None:
        print("获取彩色原始图像失败.")
    img = raw_image.convert("RGB").get_numpy_array()  # 从彩色原始图像获取RGB图像

    # 找棋盘格角点
    # 阈值
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # print(cv2.TERM_CRITERIA_EPS,'',cv2.TERM_CRITERIA_MAX_ITER)

    # w h分别是棋盘格模板长边和短边规格（角点个数）
    w = 10
    h = 7

    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵，认为在棋盘格这个平面上Z=0
    objp = np.zeros((w * h, 3), np.float32)  # 构造0矩阵，88行3列，用于存放角点的世界坐标
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 三维网格坐标划分

    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 粗略找到棋盘格角点 这里找到的是这张图片中角点的亚像素点位置，共11×8 = 88个点，gray必须是8位灰度或者彩色图，（w,h）为角点规模
    ret, corners = cv2.findChessboardCorners(gray, (w, h))

    # 如果找到足够点对，将其存储起来
    if ret:
        # 精确找到角点坐标
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        worldpoint = objp * 20  # 棋盘格的宽度为20mm (array(70,3))
        imagepoint = np.squeeze(corners)  # 将corners降为二维

        (success, rvec, tvec) = cv2.solvePnP(worldpoint, imagepoint, Camera_intrinsic_8mm["mtx"], Camera_intrinsic_8mm["dist"])

        distance = math.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2) / 10  # 测算距离
        rvec_matrix = cv2.Rodrigues(rvec)[0]
        proj_matrix = np.hstack((rvec_matrix, rvec))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]
        # 显示参数文字
        cv2.putText(img, "%.2fcm,%.2f,%.2f,%.2f" % (distance, yaw, -pitch, roll),
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imshow('findCorners', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()