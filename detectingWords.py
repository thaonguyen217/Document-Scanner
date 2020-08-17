# DOCUMENT SCANNER

import cv2
import numpy as np
import pytesseract
import os
########################################################################################################
width, height = 640, 540
color1 = (81, 57, 33) # xanh
color2 = (106, 129, 207) # hong
lang = 'eng'
path = 'Resource/paper1.jpg'
size1 = (480, 640)
size2 = (540, 720)
h_min, h_max, s_min, s_max, v_min, v_max = 0, 179, 0, 255, 180, 255
########################################################################################################

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres, imgGray, imgCanny

def getContours(img):
    contours, Hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxArea = 0
    biggest = np.array([])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 60000:
            cv2.drawContours(imgContours, cnt, -1, color2, 5)  # contour index = -1: draw all the contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > maxArea:
                maxArea = area
                biggest = approx

        cv2.drawContours(imgContours, biggest, -1, color1, 20) # contour index = -1: draw all the contours
    return biggest

def reoder(biggest):
    biggest = biggest.reshape((4,2))
    newPoints = np.zeros((4,2),np.int32)
    add = biggest.sum(axis=1)
    newPoints[0] = biggest[np.argmin(add)]
    newPoints[2] = biggest[np.argmax(add)]
    diff = np.diff(biggest,1)
    newPoints[1] = biggest[np.argmin(diff)]
    newPoints[3] = biggest[np.argmax(diff)]
    return newPoints

def wrap(img, biggest):
    if biggest.size != 0:
        biggest = reoder(biggest)
        points1 = np.float32(biggest)
        points2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(points1, points2)
        imgWrap = cv2.warpPerspective(img, matrix, (width, height))
        imgWrap = imgWrap[20:imgWrap.shape[0]-20,20:imgWrap.shape[1]-20]
    else:
        imgWrap = img
    return imgWrap

img = cv2.imread(path)
img = cv2.resize(img, size1)
# Wrap image
imgThres, imgGray, imgCanny = preProcessing(img)
imgContours = img.copy()
biggest = getContours(imgThres)
imgWrap = wrap(img, biggest)
name = path.split('/')
name1 = name[1]
path1 = os.path.join('Resource/Saved image', name1)
if os.path.exists(path1) == False:
    cv2.imwrite(path1, imgWrap)

#------------------------ APPLY TO IMAGE ------------------------

# Color detection
imgHSV = cv2.cvtColor(imgWrap, cv2.COLOR_BGR2HSV)
lower = np.array([h_min, s_min, v_min])
upper = np.array([h_max, s_max, v_max])
mask = cv2.inRange(imgHSV, lower, upper)
imgResult = cv2.bitwise_and(imgWrap, imgWrap, mask=mask)

# Text detection
imgText = cv2.cvtColor(imgWrap, cv2.COLOR_BGR2RGB)
string = pytesseract.image_to_string(imgText)
print('String: ', string)

name2 = name1.split('.')
name2 = name2[0] + '.txt'
path2 = os.path.join('Resource/Saved text', name2)
if os.path.exists(path2) == False:
    f = open(path2, 'x')
    f.write(string)
    f.close()

# DETECTING WORDS
hImg, wImg, _ = imgText.shape
# if you only want to regconize numbers:
# cong = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_data(imgText, lang='eng', config=cong)
boxes = pytesseract.image_to_data(imgText, lang=lang)
for i, box in enumerate(boxes.splitlines()):
    box = box.split()
    # print(box)
    if i != 0 and len(box) == 12:
        x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
        cv2.rectangle(imgResult, (x, y), (w+x, h+y), color=(225, 225, 0), thickness=1)
        # cv2.putText(imgResult, box[11], (x, y + 30), cv2.FONT_ITALIC,
                    # fontScale=0.4, color=(0, 0, 0), thickness=1)

# Display
imgStack = stackImages(0.6, ([img, imgGray, imgCanny, imgThres], [imgContours, imgWrap, mask, imgResult]))
cv2.imshow('Work flow', imgStack)
imgResult = cv2.resize(imgResult, size2)
cv2.imshow('Result', imgResult)
cv2.waitKey(0)