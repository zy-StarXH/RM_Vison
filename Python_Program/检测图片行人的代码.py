import cv2

img = cv2.imread('xingren2.jpg')
body = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_fullbody.xml')#加载联级分类器
bodys = body.detectMultiScale(img,1.1)#一种简写的方法 前面已经使用body进行联级分选 下面就可以直接
#使用body来detect 缩放1.15倍，最小目标尺寸为4
#绘制方框：
for (x,y,w,h) in bodys:#bodys能返回四个值 但有很多向量
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('img',img)
cv2.waitKey()#无限等待 直到键盘有任何敲击
cv2.destroyAllWindows()
