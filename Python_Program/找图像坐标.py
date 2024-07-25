import cv2
import numpy as np

# 初始化变量
click_event = False
click_position = None


# 鼠标回调函数
def mouse_event(event, x, y, flags, param):
    # 鼠标左键按下时触发
    if event == cv2.EVENT_LBUTTONDOWN:
        global click_event, click_position
        click_event = True
        click_position = (x, y)


image = cv2.imread('jinkuangshi.jpg')  # 替换为你的图片路径
resized_img = cv2.resize(image, (500, 500))
# 创建一个窗口并设置鼠标回调函数
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_event)

while True:
    # 显示图像
    cv2.imshow('image', resized_img)

    # 检查是否有按键按下或鼠标点击
    key = cv2.waitKey(1) & 0xFF

    # 如果点击了鼠标左键，则输出坐标并退出循环
    if click_event:
        print(f"Mouse clicked at ({click_position[0]}, {click_position[1]})")
        click_event = False  # 重置点击事件
        break

        # 如果按下 'q' 键，则退出循环
    if key == ord('q'):
        break

    # 关闭窗口
cv2.destroyAllWindows()