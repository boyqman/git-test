# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:29:06 2023

@author: nguye
"""

import imutils
import cv2
import numpy as np

def empty(a):
        pass

# create new window with trackbar for HSV Color

cv2.namedWindow("Range HSV")
cv2.resizeWindow("Range HSV", 500, 350)
cv2.createTrackbar("HUE Min", "Range HSV", 0,180,empty)
cv2.createTrackbar("HUE Max", "Range HSV", 180,180,empty)
cv2.createTrackbar("SAT Min", "Range HSV", 0,255,empty)
cv2.createTrackbar("SAT Max", "Range HSV", 255,255,empty)
cv2.createTrackbar("VALUE Min", "Range HSV", 0,255,empty)
cv2.createTrackbar("VALUE Max", "Range HSV", 255,255,empty)

# read image
video = cv2.VideoCapture(0)

while True:
        image = cv2.imread("Pictures/t5.jpg")
        image = imutils.resize(image, width=600)
        # image = image[0:500,230:420] 
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # get value from trackbar
        h_min = cv2.getTrackbarPos("HUE Min", "Range HSV")
        h_max = cv2.getTrackbarPos("HUE Max", "Range HSV")
        s_min = cv2.getTrackbarPos("SAT Min", "Range HSV")
        s_max = cv2.getTrackbarPos("SAT Max", "Range HSV")
        v_min = cv2.getTrackbarPos("VALUE Min", "Range HSV")
        v_max = cv2.getTrackbarPos("VALUE Max", "Range HSV")

        # define range of some color in HSV

        lower_range = np.array([h_min,s_min,v_min])
        upper_range = np.array([h_max, s_max, v_max])

        # convert image to HSV

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # threshold the hsv image to get some color

        thresh = cv2.inRange(hsv, lower_range, upper_range)

        # bitwise AND mask and original image

        bitwise = cv2.bitwise_and(image, image, mask=thresh)

        cv2.imshow("Original Image", image)
        cv2.imshow("Thresholded", thresh)
        cv2.imshow("Bitwise", bitwise)

        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:
            break
cv2.destroyAllWindows()