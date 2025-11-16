import numpy as np
import cv2

img = cv2.imread("C:\\Users\\l1221\\Documents\\GitHub\\GelSight-homemade\\Pytorch-UNet\\pred.png")
k = np.argmax(img)
pass