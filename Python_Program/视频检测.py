import cv2

capture = cv2.VideoCapture(0)
while (capture.isOpened()):
    retval,img = capture.read()
    #把视频图像转换为灰度图像
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #二值化处理灰度图
    t,binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    cv2.imshow('gray',img_gray)
    cv2.imshow('binary',binary)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img,[x,y],[x + w,y + h],(0,255,0),1)

    key = cv2.waitKey(1)
    cv2.imshow('img', img)
    if key == 32:
        cv2.imwrite('D:/pythonProject/jietu.png', img)
        break
capture.release()
cv2.destroyAllWindows()