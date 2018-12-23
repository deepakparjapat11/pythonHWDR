import numpy as np
import cv2
from preapare import x_cord_contour

from preapare import makeSquare
from preapare import resize_to_pixel
#from preapare import knn
from training import knn
image = cv2.imread('img.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image",image)
cv2.imshow("gray",gray)
cv2.waitKey(0)


blurred = cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow("blurred",blurred)
cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("edged",edged)
cv2.waitKey(0)

contours, _=cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


contours=sorted(contours, key = x_cord_contour ,reverse = False)



full_number = []

for c in contours:
    (x,y,w,h) = cv2.boundingRect(c)

    cv2.drawContours(image, contours, -1,(0,255,0), 3)
    cv2.imshow("conturs",image)

    if w>=5 and h>=25:
        roi = blurred[y:y + h, x:x + w]
        ret, roi =cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
        squared = makeSquare(roi)
        final = resize_to_pixel(20, squared)
        cv2.imshow("final",final)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        ret, result, neighbours, dist= knn.find_nearest(final_array, k=1)
        
        number =str(int(float(result[0])))
        full_number.append(number)
        cv2.rectangle(image,(x,y),(x+w, y+h),(0,0,255),2)
        
        cv2.putText(image,number,(x, y+155),
                    cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
        cv2.imshow("image",image)
        cv2.waitKey(0)


cv2.destroyAllWindows()
print("the number is:" + ''.join(full_number))

