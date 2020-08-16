# DOCUMENT SCANNER

import cv2
import numpy as np
import pytesseract
########################################################################################################
width=540
height =640
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

def getContours(img):
    contours, Hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxArea = 0
    biggest = np.array([])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 60000:
            cv2.drawContours(imgContours, cnt, -1, (255, 0, 255), 30)  # contour index = -1: draw all the contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > maxArea:
                maxArea = area
                biggest = approx

        cv2.drawContours(imgContours, biggest, -1, (255, 255, 0), 200) # contour index = -1: draw all the contours
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

########################### USING IMAGE ###########################
img = cv2.imread('Resource/cardVisit.jpg')

# Wrap image
imgContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), sigmaX=1, sigmaY=1)
imgCanny = cv2.Canny(imgBlur, 50, 100)
biggest = getContours(imgCanny)
imgWrap = wrap(img, biggest)

def empty(a):
    pass

#------------------------ COLOR PICKER ------------------------
cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars', 640, 240)
cv2.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
cv2.createTrackbar('Hue Max', 'TrackBars', 179, 179, empty)
cv2.createTrackbar('Sat Min', 'TrackBars', 0, 255, empty)
cv2.createTrackbar('Sat Max', 'TrackBars', 255, 255, empty)
cv2.createTrackbar('Val Min', 'TrackBars', 0, 255, empty)
cv2.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)

while True:
    h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')
    h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')
    s_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')
    s_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')
    v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')
    v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')
    imgHSV = cv2.cvtColor(imgWrap, cv2.COLOR_BGR2HSV)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(imgWrap, imgWrap, mask=mask)

    imgText = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(imgText))

    imgStack = stackImages(0.2, ([img, imgCanny, imgContours], [imgWrap, mask, imgResult]))
    cv2.imshow('Work flow', imgStack)
    cv2.waitKey(1)



