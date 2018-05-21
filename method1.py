from darkflow.net.build import TFNet
import cv2
import os 
from math import pow
import time

options= {"model" : "cfg/yolo.cfg", 
            "load" : "bin/yolo.weights", 
            "threshold" : 0.15}

tfnet = TFNet(options)

y_stardard = 275
count_range = 10
distance_standard = 10000

length = len(os.listdir('test_image5'))
font = cv2.FONT_HERSHEY_SIMPLEX

counter = 0
start_time = time.time()
for index in range(1, length+1):
    print('%d/%d' % (index,length))
    centers = []
    false_centers = []
    true_centers=[]
    filename = 'test_image5/image_' + str(index) + '.jpg'
    image = cv2.imread(filename)
    result = tfnet.return_predict(image)
    print(result)
    for det in result:
        if det['label'] == 'car' or det['label'] == 'cell phone':
            x_center = (det['topleft']['x'] + det['bottomright']['x'])/2
            y_center = (det['topleft']['y'] + det['bottomright']['y'])/2
            topleft = (det['topleft']['x'], det['topleft']['y'])
            bottomright = (det['bottomright']['x'], det['bottomright']['y'])
            centers.append([x_center, y_center, topleft, bottomright])
    print(centers)
    if len(centers) <= 1:
        pass 
    else:
        for i in range(len(centers)-1):
            for j in range(i+1,len(centers)):
                distance =pow(centers[i][0] - centers[j][0], 2) + pow(centers[i][1] -centers[j][1], 2)
                if distance < distance_standard:
                    false_centers.append(centers[j])
    print(false_centers)
    if false_centers == []:
        true_centers = centers 
    else:
        for center in centers:
                if not center in false_centers:
                    true_centers.append(center)
    print(true_centers) 
    for true_center in true_centers:
        if abs(true_center[1] - y_stardard) < count_range:
            counter += 1
            cv2.rectangle(image, true_center[2], true_center[3], (0, 255, 0), 4)
        else:
            cv2.rectangle(image, true_center[2], true_center[3], (255, 0, 0), 4)
    cv2.putText(image, str(counter), (100, 100), font, 2, (0, 0, 255), 2)
    cv2.imwrite('test_labeled/image_' + str(index) +'.jpg', image)
    print(counter)

end_time = time.time()
average = (end_time - start_time) / length
print('average time is %.4f' % average)

