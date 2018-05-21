import cv2
import time

start = time.time()
cap = cv2.VideoCapture('test.mp4')
count = 0
timeF = 1
rval = cap.isOpened()
while rval:
    count += 1
    rval,frame = cap.read()
    if rval:
        cv2.imwrite('1_image/test_image/image_'+str(count)+'.jpg', frame)
        cv2.waitKey(1)
    else:
        break
cap.release()
print((time.time() - start) / count)
