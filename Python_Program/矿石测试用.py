import cv2

img = cv2.imread('refix2.jpg')

blurred = cv2.GaussianBlur(img, (5, 5), 0)
resized_img = cv2.resize(blurred, (500, 500))

gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
blurred = cv2.GaussianBlur(binary, (5, 5), 0)
contours, _ = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(resized_img, contours, -1, (255, 0, 0), 2)
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(f"轮廓 {i+1} 的面积是: {area}")
cv2.imshow('img',resized_img)
cv2.imshow('gray_img', blurred)
cv2.waitKey()
cv2.destroyAllWindows()
# 黑色正方形面积为5225， 三角梯形面积为2873 4180