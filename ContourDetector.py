import numpy as np
import cv2

def contour():
    img = cv2.imread('TrainDB/T_jh01_withMany.jpeg')
    height, width = img.shape[:2]
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray,0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr  = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

    contours, _=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1,(0,0,255),1)
    cv2.imshow('thresh',thr)
    # cv2.imshow('contour',img)

    cv2.waitKey(0)
    cv2.destroyWindows()

contour()
