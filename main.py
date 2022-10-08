import cv2
import numpy as np
from utils import *

# ---
img_path = "Resources/sudoku2.jpg"
img_widht, img_height = 810, 810

# ---
# preprocess Image
img = cv2.resize(cv2.imread(img_path), (img_widht, img_height))

img_processed = cv2.adaptiveThreshold(cv2.GaussianBlur(
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1), 255, 1, 1, 11, 2)

# find contours

img_contours = img.copy()
contours, hierarchy = cv2.findContours(
    img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

# find biggest contour
img_biggestContour = img.copy()
biggest_contour, biggest_area = defineBiggestContour(contours)
if biggest_contour.size > 0:
    biggest_contour = orderPoints(biggest_contour)
    cv2.drawContours(img_biggestContour, biggest_contour, -1, (255, 0, 0), 20)

# warp perspective
    pts1 = np.float32(biggest_contour)
    pts2 = np.float32(
        [[0, 0], [img_widht, 0], [0, img_height], [img_widht, img_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_soduko = cv2.warpPerspective(
        img, matrix, (img_widht, img_height))
    img_soduko_black = cv2.cvtColor(img_soduko, cv2.COLOR_BGR2GRAY)

# read the soduko
all_fields = []
rows = np.vsplit(img_soduko_black, 9)
for i in rows:
    columns = np.hsplit(i, 9)
    for j in columns:
        all_fields.append(j)

# for i in all_fields:
   # pytesseract.image_to_string(i)


# output
cv2.imshow("img", img)
cv2.waitKey(0)

cv2.imshow("processed", img_processed)
cv2.waitKey(0)

cv2.imshow("contours", img_contours)
cv2.waitKey(0)

cv2.imshow("biggest contour", img_biggestContour)
cv2.waitKey(0)

cv2.imshow("warp perspective", img_soduko)
cv2.waitKey(0)

cv2.imshow("warp perspective black", img_soduko_black)
cv2.waitKey(0)

for i in all_fields:
    cv2.imshow("single field", i)
    cv2.waitKey(0)
