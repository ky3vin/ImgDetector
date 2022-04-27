import cv2
import numpy as np
import os

path='ImageDB'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)