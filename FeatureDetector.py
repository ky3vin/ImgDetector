import cv2
import numpy as np

# 비교에 사용할 이미지 불러오기(gray scale)
img1 = cv2.imread('ImageDB/jh03.jpeg', 0)
img2 = cv2.imread('TrainDB/T_jh03.jpeg', 0)

# ORB객체 생성, ORB 객체가 한 번에 검출하고자 하는 특징점의 개수는 5000개
# cv2.ORB_create(최대 피처 수, 스케일 계수, 피라미드 레벨, 엣지 임곗값, 시작 피라미드 레벨, 비교점, 점수 방식, 패치 크기, FAST 임곗값)
orb = cv2.ORB_create(nfeatures=5000)

# 특징점 및 기술자 계산 메서드(orb.detectAndCompute)로 각각의 이미지에서 특징점 및 기술자를 계산
# 특징점, 기술자 = orb.detectAndCompute(입력 이미지, 마스크)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 기술자 : 각 특징점을 설명하기 위한 2차원 배열로, 이 배열은 두 특징점이 같은지 판단할 때 사용됨
print(des1.shape)
print(des1[0])

# 검출한 Keypoints가 어느 위치에 존재하는지 확인하기 위해 그리기
imgKp1 = cv2.drawKeypoints(img1, kp1, None)
imgKp2 = cv2.drawKeypoints(img2, kp2, None)

# 전수 조사 매칭 클래스(cv2.BFMatcher)로 전수 조사 매칭을 사용
# 전수 조사 매칭은 객체의 이미지와 객체가 포함된 이미지의 각 특징점을 모두 찾아 기술자를 활용
bf = cv2.BFMatcher()

# 첫 번째 파라미터인 queryDescriptors를 기준으로 두 번째 파라미터인 trainDescriptors에 맞는 매칭을 찾음
# 가장 비슷한 k개만큼의 매칭 값을 반환(k: 매칭할 근접 이웃 개수)
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# 매칭점을 이미지에 표시
# img1, kp1: queryDescriptor의 이미지와 특징점
# img2, kp2: trainDescriptor의 이미지와 특징점
# matches: 매칭 결과
# flags: 매칭점 그리기 옵션 (cv2.DRAW_MATCHES_FLAGS_DEFAULT: 결과 이미지 새로 생성(default값)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imshow('img3',img3)
cv2.waitKey(0)