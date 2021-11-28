import cv2
import numpy as np
from sklearn.linear_model import LinearRegression as LR

sum_list = []
for image_number in range(1, 31):
    if image_number < 10:
        img = cv2.imread(f'./Open/t0{image_number}/5.jpg')
    else:
        img = cv2.imread(f'./Open/t{image_number}/5.jpg')

    img = cv2.resize(img, dsize=(0,0), fx = 0.25, fy = 0.25, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 700, param1=50, param2=30, minRadius=300, maxRadius=400)

    if len(circles) == 1:
        x, y, r = map(int, circles[0][0])
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1, 8, 0)
        out = gray * mask
        sum = 0
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                sum += out[i][j]
        sum_list.append(sum)
    else:
        sum_list.append('error')

amount = np.array([44,60,49,129,39,22,513,196,263,170,98,379,1600,2,31,5,1190,151,108,122,75,84,10,24,1375,7,1032,1429,1323,691]).reshape(-1, 1)
value = np.array(sum_list).reshape(-1, 1)

lm = LR()
lm.fit(amount, value)

print(lm.intercept_)
print(lm.coef_)