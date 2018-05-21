from darkflow.net.build import TFNet
import cv2
import os 
import time 
import numpy as np
import json

options = {"model": "cfg/yolo.cfg", 
        "load": "bin/yolo.weights", 
        "threshold":0.15}

tfnet = TFNet(options)

distance_standard = 10000

image_paths = [
        #'test_image1', 
        #'test_image2', 
        #'test_image3', 
        #'test_image4',
        #'test_image5',
        'test_image6',
        #'test_image7',
        #'test_image8'
        ]

start = time.time()
for image_path in image_paths:
    length = len(os.listdir('1_image/' + image_path))
    f = open('4_predict/' + image_path + '.txt', 'w')

    for index in range(1, length+1):
        print('%d/%d' % (index, length))
 
        car_centers = []
        cell_centers = []
        false_centers = []
        true_centers=[]

        filename = '1_image/' + image_path + '/image_' + str(index) + '.jpg'
        image = cv2.imread(filename)
        result = tfnet.return_predict(image)
        for det in result:
            if det['label'] == 'car':
                x_center = (det['topleft']['x'] + det['bottomright']['x'])/2
                y_center = (det['topleft']['y'] + det['bottomright']['y'])/2
                topleft = (det['topleft']['x'], det['topleft']['y'])
                bottomright = (det['bottomright']['x'], det['bottomright']['y'])
                car_centers.append([x_center, y_center, topleft, bottomright])
            elif det['label'] == 'cell phone':
                x_center = (det['topleft']['x'] + det['bottomright']['x'])/2
                y_center = (det['topleft']['y'] + det['bottomright']['y'])/2
                topleft = (det['topleft']['x'], det['topleft']['y'])
                bottomright = (det['bottomright']['x'], det['bottomright']['y'])
                cell_centers.append([x_center, y_center, topleft, bottomright])

        print('car_centers:', car_centers)
        print('cell_centers:', cell_centers)

        if len(cell_centers):
            for i in range(len(cell_centers)):
                for j in range(len(car_centers)):
                    distance = pow(cell_centers[i][0] - car_centers[j][0], 2)+ pow(cell_centers[i][1] - car_centers[j][1], 2)
                    if distance < distance_standard:
                        false_centers.append(cell_centers[i])
        print('false_centers', false_centers)

        for center in cell_centers:
            if not center in false_centers:
                car_centers.append(center)
        print('car_centers:', car_centers)

        for center in car_centers:
            if center[3][0] - center[2][0] < 500:
                f.write(json.dumps(center))
                f.write(':')
        f.write('\n')

    f.close()

end = time.time()

print('cost time:', end - start)


