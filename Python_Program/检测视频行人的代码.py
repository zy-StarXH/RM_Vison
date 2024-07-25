import cv2
body = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_fullbody.xml')

capture = cv2.VideoCapture(0)
while (capture.isOpened()):
    retval,img = capture.read()
    bodys = body.detectMultiScale(img,1.15)
    for (x,y,w,h) in bodys:#bodys能返回四个值 但有很多向量
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('img',img)
    key = cv2.waitKey(1)
    if key == 32:  # 如果检测到按下空格键
        break
cv2.destroyAllWindows()
capture.release()
