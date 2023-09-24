# USAGE
# python ocr_handwriting.py --model handwriting.model --image images/umbc_address.png

# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# construct the argument parser and parse the arguments

def find_contours(image):
    cnts= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cnts = sorted(cnts,key=lambda b:b[1][0], reverse = False)    

    return cnts

# load the handwriting OCR model
print("[INFO] loading handwriting OCR model...")
model = load_model("char.model")




# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
img = cv2.imread("Pictures/1.jpg")
img = imutils.resize(img, width=600)
#cv2.imshow("Image_org",img)
blur = cv2.GaussianBlur(img, (5,5),0)  

lower_white=np.array([110,0,150])  #hsv
upper_white=np.array([180,60,255])
# lower_white=np.array([0,0,153])  #hsv
# upper_white=np.array([180,45,255])

lower_black=np.array([0,0,0])  #hsv
upper_black=np.array([180,255,140])

hsv_img=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
mask_white=cv2.inRange(hsv_img,lower_white,upper_white)
 
cnts_white =  find_contours(mask_white)
chars = []
for cw in cnts_white:  
    if cv2.contourArea(cw)>1000:
        (x,y,w,h) = cv2.boundingRect(cw)
        Cropped_white = mask_white[y:y+h, x:x+w]
        amount_white=cv2.countNonZero(Cropped_white)
        
        if h > 30 and h < 78  and w >160 and w < 390 and amount_white >3000:
            img_crop = img[y:y+h, x:x+w]
            blur_crop = blur[y:y+h, x:x+w]
            hsvimg_crop = hsv_img[y:y+h, x:x+w]

            mask_black=cv2.inRange(hsvimg_crop,lower_black,upper_black)
            cnts_black = find_contours(mask_black)
            for cb in cnts_black: 
                (x1,y1,w1,h1) = cv2.boundingRect(cb)
                if h1 > 20 and h1 < 60  and w1 >15 and w1 < 57:
                    print(h,w)
                    char_crop = blur_crop[y1:y1+h1, x1:x1+w1]
                    #cv2.rectangle(img_crop, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                    
                    char_crop = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)

                    char_crop = cv2.resize(char_crop, (32, 32))
                    char_crop = img_to_array(char_crop)
                    
                    
                    padded = char_crop.astype("float32") / 255.0
                    padded = np.expand_dims(padded, axis=-1)
                    
                    chars.append((padded, (x+x1, y+y1, w1, h1)))

            		
# cv2.imshow("Image_crop",img_crop)
# extract the bounding box locations and padded characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")

# OCR the characters using our handwriting recognition model
preds = model.predict(chars)

# define the list of label names
labelNames = "#0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# loop over the predictions and bounding box locations together
for (pred, (x, y, w, h)) in zip(preds, boxes):
 	# find the index of the label with the largest corresponding
 	# probability, then extract the probability and label
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]

 	# draw the prediction on the image
    print("[Prediction] {} - {:.2f}%".format(label, prob * 100))
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(img, label, (x, y+h+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255), 2)

cv2.imshow("mask_black",mask_black)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()