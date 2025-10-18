import cv2 as cv
import numpy as np

img = cv.imread("hello.jpeg")

mat = img
height = np.size(img, 0)
width = np.size(img, 1)

bwImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edge = cv.Canny(bwImg, 100, 150)

# print(mat)
# print(height, width)

# cv.imwrite("gray.png", bwImg)
# cv.imshow("Original Image", img)
cv.imshow("Gray Image", bwImg)
cv.imshow("Edge", edge)
k = cv.waitKey(0)