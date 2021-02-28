#include <opencv.hpp>
import cv2
import numpy as np
from PIL import Image

original_img = cv2.imread('alph.png')
original_img=original_img[int(len(original_img)/5)*4:int(len(original_img)/5)*5,:]
B, G, R = cv2.split(original_img)                    #get the bull channel (because the line is blue)
img = original_img
line=13
_,cleanBlue = cv2.threshold(B,100,255,cv2.THRESH_BINARY)  #get rid of blue line by setting
ridfuzz= cv2.dilate(cleanBlue,None,iterations=1)  #erode to get rid of fuzz
ridfuzz2= cv2.erode(ridfuzz,None,iterations=1)
erode = cv2.erode(ridfuzz2,None,iterations=2)        #dialate to fill the gap caused by blue line
dilated = cv2.dilate(erode,None,iterations=2)      #膨胀图像
res = cv2.resize(dilated,(300*line, 300),
                 interpolation = cv2.INTER_CUBIC) ##turn the image into 10*16*16
       #cv2.imshow("original_img", res)             #
#cv2.imshow("B_channel_img", img)            #
#cv2.imshow("RidBlue", cleanBlue)          #
#cv2.imshow("ridfuzz2", ridfuzz2)          #
#cv2.imshow("erode", erode)          #
cv2.imshow("Dilated Image",dilated)         #
cv2.imshow("Image",res)         #
cv2.waitKey(0)
cv2.destroyAllWindows()
print(len(res))
print(len(res[0]))
for i in range(len(res[0])):        #turn the picture binary again
	for j in range(len(res)):
		if res[j][i]>=100:
			res[j][i]=255
		else:
			res[j][i]=0
print(res)
print(res[10])
line=13
for i in range(line):
	a=res[:,int(len(res[0])/line)*i:int(len(res[0])/line)*(i+1)]
	name=str(i+line*4) + ".png"
	cv2.imwrite(name,a)