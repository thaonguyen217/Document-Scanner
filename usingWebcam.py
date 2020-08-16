# DOCUMENT SCANNER

import cv2
import numpy as np
import pytesseract
import os
########################################################################################################
width = 540
height = 640
########################################################################################################
def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres

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
        if area > 10000:
            cv2.drawContours(imgContours, cnt, -1, (255, 0, 255), 30)  # contour index = -1: draw all the contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > maxArea:
                maxArea = area
                biggest = approx
                x, y, w, h = cv2.boundingRect(approx)

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


########################### USING WEBCAM ###########################
cap = cv2.VideoCapture(0)
cap.set(3,640)  # width (id:3) = 640
cap.set(4,480)  # height (id:4) = 480
cap.set(10,50) # brightness (id:10) = 100
count = 0

while True:
    success, img = cap.read()
    # Wrap image
    imgContours = img.copy()
    imgThres = preProcessing(img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), sigmaX=1, sigmaY=1)
    imgCanny = cv2.Canny(imgBlur, 50, 100)
    biggest = getContours(imgThres)
    imgWrap = wrap(img, biggest)

    # Color detection
    imgHSV = cv2.cvtColor(imgWrap, cv2.COLOR_BGR2HSV)
    h_min, h_max, s_min, s_max, v_min, v_max = 0, 179, 0, 255, 0, 255
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(imgWrap, imgWrap, mask=mask)

    # Text detection
    imgText = cv2.cvtColor(imgWrap, cv2.COLOR_BGR2RGB)
    string = pytesseract.image_to_string(imgText)
    print('String: ', string)

    # DETECTING CHARACTER:
    hImg, wImg, _ = imgText.shape
    boxes = pytesseract.image_to_boxes(imgText, lang='eng')
    for box in boxes.splitlines():
        box = box.split(' ')
        x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        cv2.rectangle(imgText, (x, hImg - y), (w, hImg - h), color=(255, 255, 0), thickness=1)
        # cv2.putText(imgText, box[0], (x, hImg - y + 10), cv2.FONT_ITALIC, fontScale=0.4, color=(0, 0, 0), thickness=1)

        imgStack = stackImages(0.6, ([img, imgWrap], [imgResult, imgText]))
        cv2.imshow('Work flow', imgStack)

    # Save image and text
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('Resource/Scanned/Saved image/image_' + str(count) + '.jpg', imgResult)
        f = open('Resource/Scanned/Saved text/image_' + str(count) + '.txt', 'x')
        f.write(string)
        f.close()
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
                    2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
