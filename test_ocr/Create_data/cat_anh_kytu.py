import cv2
import imutils
import numpy as np
import pytesseract

def find_contours(image):
    cnts= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)    
    screenCnt = None
    return cnts

dem = 0
img = cv2.imread("Pictures/2.jpg")
img = imutils.resize(img, width=600)
blur = cv2.GaussianBlur(img, (5,5),0)  
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

lower_white=np.array([110,0,150])  #hsv
upper_white=np.array([180,60,255])

# lower_white=np.array([0,0,175])  #hsv
# upper_white=np.array([180,30,255])


lower_black=np.array([0,0,0])  #hsv
upper_black=np.array([180,255,140])

hsv_img=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)



mask_white=cv2.inRange(hsv_img,lower_white,upper_white)
cv2.imshow("abc",mask_white)
cnts_white =  find_contours(mask_white)


for cw in cnts_white:  
    if cv2.contourArea(cw)>100:
        (x,y,w,h) = cv2.boundingRect(cw)
        
        Cropped_white = mask_white[y:y+h, x:x+w]

        cv2.rectangle(img,(x,y), (x+w,y+h), (0, 255, 0), 1)
        
        print(h,w)

        if h > 10 and h < 120  and w >160 and w < 480:
            
            img_crop = img[y:y+h, x:x+w]
            cv2.imshow("img_crop",img_crop)
            
            gray_crop = gray[y:y+h, x:x+w]
            
            hsvimg_crop = hsv_img[y:y+h, x:x+w]
            cv2.imshow("hsvimg_crop",hsvimg_crop) 
            
            mask_black=cv2.inRange(hsvimg_crop,lower_black,upper_black)
            cv2.imshow("mask_black",mask_black) 
            
            
            #cv2.imshow("mask_black", mask_black)
            cnts_black = find_contours(mask_black)
            print(h,w)
            for cb in cnts_black: 
                (x,y,w,h) = cv2.boundingRect(cb)
                if h > 20 and h < 80  and w >10 and w < 50:
                    print(h,w)
                    dem = dem+1
                    
                    char_crop = gray_crop[y:y+h, x:x+w]
                    cv2.imwrite('char/t36hfhfhmgj6'+str(dem)+'.jpg',char_crop)
                    
                    cv2.rectangle(img_crop,(x,y), (x+w,y+h), (0, 255, 0), 1)
                    cv2.imshow("img_crop", img_crop)

cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()








