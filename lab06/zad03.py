import cv2
import numpy as np
import os

images = os.listdir('c:/Users/maria/Desktop/ug/sem4/IntelegencjaObliczeniowa/lab06/bird_miniatures')
for image_name in images:
    image = cv2.imread('c:/Users/maria/Desktop/ug/sem4/IntelegencjaObliczeniowa/lab06/bird_miniatures/' + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,11,2)
    image = cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=1)
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    s1 = 8
    s2 = 1000
    xcnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > s1 and cv2.contourArea(cnt) < s2:
            xcnts.append(cnt)
    print(f'{image_name}: {len(xcnts)} birds.')