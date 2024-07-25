import cv2
import numpy as np

a1 = cv2.imread('kuangshi1.png')
a2 = cv2.imread('kuangshi2.png')
a3 = cv2.imread('kuangshi3.png')

gray1 = cv2.cvtColor(a1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(a2,cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(a3,cv2.COLOR_BGR2GRAY)

_, b1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
_, b2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
_, b3 = cv2.threshold(gray3, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('D:/pythonProject/kuangshia.png',b1)
cv2.imwrite('D:/pythonProject/kuangshib.png',b2)
cv2.imwrite('D:/pythonProject/kuangshic.png',b3)

cv2.destroyAllWindows()