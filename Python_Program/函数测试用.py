import cv2
import numpy as np

# 初始化标记个数：
left_arrow_sign = 0
left_square_sign = 0
right_arrow_sign = 0
right_square_sign = 0
rd_arrow_sign = 0
rd_square_sign = 0
ld_arrow_sign = 0
ld_square_sign = 0
posture = 'normal'
ip_address = '10.13.81.95'  # 使用droidcam调用手机摄像头，应自行更改ip地址
port = 4747  # 端口
# capture = cv2.VideoCapture(f'http://{ip_address}:{port}/mjpegfeed')  # 使用网站调用摄像头
capture = cv2.VideoCapture('target_video10.mp4')
# 定义黄色范围：
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
silver_lower = np.array([0, 0, 50])
silver_upper = np.array([180, 255, 255])

# 获取输入：
user_input = input("键入a检测金矿石，键入b检测银矿石：")


# 提取画面中的黄色部分，找到金矿石的大致位置：
def find_yellow_area(image):
    resized_img = cv2.resize(image, (500, 500))
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)  # 创建掩模
    yellow_img = cv2.bitwise_and(resized_img, resized_img, mask=mask)
    yellow_img[np.where(mask == 0)] = [0, 0, 0]  # 将非黄色区域设置为黑色
    gray = cv2.cvtColor(yellow_img, cv2.COLOR_BGR2GRAY)
    s, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def gain_roi():
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

while True:
    ret, image = capture.read()
    if user_input == "a":
        find_yellow_area(image)
        contours = find_yellow_area(image)
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