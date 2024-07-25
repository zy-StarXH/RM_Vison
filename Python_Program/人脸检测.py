import cv2

#加载识别人脸的级联分类器
faceCascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')#加载识别眼镜的联级分类器

capture = cv2.VideoCapture(0)
while (capture.isOpened()):
    retval, img = capture.read()#retval返回值一般没用
    faces = faceCascade.detectMultiScale(img, 1.15)#识别联级分类器的内容
    for (x,y,w,h) in faces:#获得的左上角坐标，宽 高
        cv2.rectangle(img,(x,y),(x + w,y + h),(0,255,0),1)#绘制方框
    cv2.imshow('img',img)
    key = cv2.waitKey(1)#停顿1ms后进入下一次循环
    if key == 32:#如果检测到按下空格键
        break
cv2.destroyAllWindows()
capture.release()


