import cv2 as cv
import numpy as np


PHOTOS = r"C:\Users\rowe1\Desktop\git\miscellaneous\learning open cv\opencv-course-master\Resources\Photos"
FILE = "images/p5mask-gzwop-imgbin.png"
img = cv.imread(FILE, cv.IMREAD_UNCHANGED)
blank = np.zeros(img.shape[:2], dtype='uint8')
color = np.full(img.shape[:2], 255, dtype='uint8')
background = cv.merge([blank, color, blank, blank])
face = cv.imread(f'{PHOTOS}/lady.jpg')


#cv.imshow('blank', background)
#cv.imshow('mask', img)
#cv.imshow('overlay', background + img)
#cv.imshow('alpha', img[:, :, 3])

#cv.imshow('original', img)
mask = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
#cv.imshow('resized', img2)


alpha = mask[:,:,3]
m = mask[:,:,3] > 10

x = 100
y = 200
subface = face[y:y+mask.shape[0], x:x+mask.shape[1]]

#face[y:y+mask.shape[0], x:x+mask.shape[1]] = mask[mask[:,:,3] > 10]


#for i in range(mask.shape[0]):
#    for j in range(mask.shape[1]):
#        if mask[i][j][3] > 10:
#            face[y+i][x+j] = mask[i][j][:3]
        
cv.imshow('x', face)

cv.waitKey(0)
cv.destroyAllWindows()