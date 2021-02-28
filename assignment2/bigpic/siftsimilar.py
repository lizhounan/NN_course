import cv2 
import matplotlib.pyplot as plt


# read images
img1 = cv2.imread('9.png')  
img2 = cv2.imread('18.png') 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


#sift
sift = cv2.xfeatures2d.SIFT_create()
# 查找监测点和匹配符

kp1, des1 = sift.detectAndCompute(img1, None)

kp2, des2 = sift.detectAndCompute(img2, None)

print(len(kp1), len(des1) )   # 1402, 1402





FLANN_INDEX_KDTREE = 0

indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(indexParams, searchParams)

# 进行匹配

matches = flann.knnMatch(des1, des2, k=2)

# 准备空的掩膜 画好的匹配项

matchesMask = [[0, 0] for i in range(len(matches))]

 

for i, (m, n) in enumerate(matches):

	if m.distance < 0.7*n.distance:

		matchesMask[i] = [1, 0]

 

drawPrams = dict(matchColor=(0, 255, 0),

				 singlePointColor=(255, 0, 0),

				 matchesMask=matchesMask,

				 flags=0)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawPrams)

img_PutText = cv2.putText(img3, "SIFT+kNNMatch: Image Similarity Comparisonn", (40, 40),cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3,)

img4 = cv2.resize(img_PutText, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)  #缩小1/2

 

cv2.imshow("matches", img4)

cv2.waitKey(7000)

cv2.destroyAllWindows()