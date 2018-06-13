import cv2

videoWriter = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720))

for i in range(1, 10):
    image = cv2.imread('image_' + str(i) + '.jpg')
    videoWriter.write(image)
