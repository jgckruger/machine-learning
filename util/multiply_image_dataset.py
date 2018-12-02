### Author: João Gabriel Corrêa Krüger

import os
import cv2
import imutils

def mirrorAndSave(dir,name):
	img = cv2.imread(os.path.join(root,name))
	mirror = cv2.flip(img,1);
	cv2.imwrite(os.path.join(root,name[:-4:]+'_mirror'+name[-4::]),mirror)

def rotateAndSave(dir, name, angle):
	img = cv2.imread(os.path.join(root,name))
	rot = imutils.rotate_bound(img, angle)
	cv2.imwrite(os.path.join(root,name[:-4:]+'_angle'+str(angle)+name[-4::]),rot)

for root, dirs, files  in os.walk("./images"):
	for name in files:
		mirrorAndSave(root,name)

for root, dirs, files  in os.walk("./images"):
	for name in files:
		rotateAndSave(root,name,-10)
		rotateAndSave(root,name,10)
		rotateAndSave(root,name,-5)
		rotateAndSave(root,name,5)
