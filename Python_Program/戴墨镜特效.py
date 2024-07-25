import cv2

#定义覆盖图像：
def overlay_img(img,img_over,img_over_x,img_over_y):
    img_w, img_h = img.shape#原图像的宽高：
    img_over_h,img_over_w,img_over_c = img_over.shape#要覆盖的图像的宽高和通道数
    if img_over_c == 3:
        img_over = cv2.cvtColor(img_over,cv2.COLOR_BGR2RGBA)#把要覆盖的图像转换成四通道的图像
        for w in range(0,img_over_w):#遍历覆盖图像的列
            for h in range(0,img_over_h):#遍历覆盖图像的行
                if img_over[h,w,3] != 0:#如果不是全透明像素：
                    for c in range(0,3):
                        x = img_over_x + w
                        y = img_over_y + h
                        if x >= img_w or y >= img_h:
                            break
                        img[x,y,c] = img_over[h,w,c]
        return img

#加载识别人脸的级联分类器
faceCascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')

glass_img = cv2.imread('sunglasses.jpg',cv2.IMREAD_UNCHANGED)
height,width,channel = glass_img.shape

capture = cv2.VideoCapture(0)
while (capture.isOpened()):
    retval, img = capture.read()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img, 1.15)
    for (x,y,w,h) in faces:
        gw = w
        gh = int(height*w/width)
        glass_img = cv2.resize(glass_img,(gw,gh))
        overlay_img(img,glass_img,x,y + int(h*1/3))
    cv2.imshow('img',img)
    key = cv2.waitKey(1)
    if key == 32:
        break
cv2.destroyAllWindows()
capture.release()
