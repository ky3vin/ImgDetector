import cv2
import numpy as np
import os

path='ImageDB'
orb=cv2.ORB_create(nfeatures=5000)

images=[]
classNames=[]
myList=os.listdir(path)

for cl in myList:
    imgCur=cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findDes(images):
    desList=[]
    for img in images:
        kp, des=orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def findID(img, desList, thres=65):
   kp2, des2=orb.detectAndCompute(img,None)
   kp2_r, des2_r=orb.detectAndCompute(cv2.flip(img, 1),None)
   bf=cv2.BFMatcher()
   matchList=[]
   matchList_r=[]
   finalVal=-1
   try:
       for des in desList:
           matches = bf.knnMatch(des, des2, k=2)
           matches_r = bf.knnMatch(des,des2_r,k=2)
           good = []
           good_r=[]
           for m,n in matches:
               if m.distance < 0.75 * n.distance:
                  good.append([m])
           matchList.append(len(good))
           print(matchList)

           for m_r,n_r in matches_r:
               if m_r.distance < 0.75 * n_r.distance:
                  good_r.append([m_r])
           matchList_r.append(len(good_r))
           print(matchList_r)
   except:
       pass
   if len(matchList)!=0:
      if max(matchList) > thres:
          finalVal = matchList.index(max(matchList))
      elif max(matchList_r) > thres*3:
           finalVal = matchList_r.index(max(matchList_r))
   return finalVal
desList=findDes(images)

img2=cv2.imread('TrainDB/T_jh03_reverse.jpeg', cv2.COLOR_BGR2GRAY)
imgOriginal=img2.copy()
id = findID(img2, desList)
if id != -1:
    cv2.putText(imgOriginal, classNames[id], (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
cv2.imshow('img2',imgOriginal)
cv2.waitKey(0)