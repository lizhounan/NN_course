import gif2numpy
import cv2
import numpy as np
from PIL import Image
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.array(gray).astype('int')

np_frames, extensions, image_specifications = gif2numpy.convert("alphabet.gif")
img = np_frames[0]

for i in range(10):
	#cv2.imshow(chr(65 + i), img[0:44, i*43:(i+1)*43+1, :])
	cv2.imwrite(chr(48 + (i+1)%10)+'.jpg', img[43*4:43*5+1, i*43:(i+1)*43+1, :])

cv2.waitKey()
cv2.destroyAllWindows() 


