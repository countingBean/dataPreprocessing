import cv2
import numpy as np
from PIL import Image
import os

def preprocessing():
    sum_list = []
    for image_number in range(1, 31):
        if image_number < 10:
            img = cv2.imread(f'./Hidden/t0{image_number}/5.jpg')
        else:
            img = cv2.imread(f'./Hidden/t{image_number}/5.jpg')

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
    
    results = []
    for j in range(1, 5):
        for i in range(1, 31):
            if(i < 10):
                src = cv2.imread('./Hidden/t0{}/{}.jpg'.format(i, j))
            else:
                src = cv2.imread('./Hidden/t{}/{}.jpg'.format(i, j))
            resized = cv2.resize(src, dsize=(0,0), fx = 0.25, fy = 0.25, interpolation=cv2.INTER_LINEAR)
            h = resized.shape[0]
            w = resized.shape[1]
            x = int(w/2)
            y = int(h/2)

            resized = resized[y-int(h/4):y+int(h/4), x-int(w/4):x+int(w/4)]

            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            white_out = cv2.inRange(hsv, (15, 0, 210), (43, 145, 255))
            if(i < 10):
                cv2.imwrite('./result{}/0{}.png'.format(j, i), white_out)
            else:
                cv2.imwrite('./result{}/{}.png'.format(j, i), white_out)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        k = 1
        IMG_DIR = './result{}'.format(j)
        for img in os.listdir(IMG_DIR):
            img = Image.open(os.path.join(IMG_DIR,img))
            img_array = np.array(img)
            with open('./result{}_csv/{}.csv'.format(j, k), 'w') as f:
                np.savetxt(f, img_array, fmt='%d', delimiter=",")
                k+=1
                    
        result = []
        for i in range(1, 31):
            array = np.loadtxt('./result{}_csv/{}.csv'.format(j, i), delimiter = ',', dtype = 'int')
            count = 0
            for arr in array:
                num_ones = (arr == 1).sum()
                num_255s = (arr == 255).sum()
                count += num_ones
                count += num_255s
            result.append(count)
        
        results.append(result)
    feat = np.transpose(np.array(results))
    tmp = np.transpose(np.array([sum_list]))
    feat = np.concatenate((feat, tmp), axis = 1)
    
    return feat

def predict(x_li):
    coef_li = [-6.02220618e-03,1.06968734e-02,2.21735495e-02,4.40740074e-04,3.50439474e-06]
    y = 0.0
    for i in range(len(x_li)):
        y += x_li[i] * coef_li[i]
    
    y += -226.1979423511055
    if(y < 0):
        y = 10
    return int(y)

import time
import datetime

def main():
    y_li = []
    x_li = preprocessing()
    for x in x_li:
        yi = predict(x)
        y_li.append(yi)
    return y_li

def setOutput(t, pred):
    now = datetime.datetime.now()
    f = open('Kong_10.txt', 'w')
    f.write('%TEAM   {}\n'.format('first'))
    f.write('%DATE   {}\n'.format(now.strftime('%d-%H-%M-%S')))
    f.write('%TIME   {0:.1f}\n'.format(t))
    f.write('%CASES  {}\n'.format(len(pred)))
    for i in range(1,len(pred)+1):
        f.write(f"T{i:02}     {pred[i-1]}\n")



startT = time.time()
arr = main()
endT = time.time()
setOutput(endT - startT, arr)