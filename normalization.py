import json
import cv2
import os
cls = 0
def save_to_yolo_format(cls, bbox, img_width, img_height, output_path):
    x_center = round((bbox[0][0] + bbox[1][0]) / 2.0 / img_width,6)
    y_center = round((bbox[0][1] + bbox[1][1]) / 2.0 / img_height,6)
    width = round((bbox[1][0] - bbox[0][0]) / img_width,6)
    height = round((bbox[1][1] - bbox[0][1]) / img_height,6)
    
    with open(output_path, 'w') as f:
        f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

b = '라벨링데이터/TL_01.고라니/'
files = os.listdir(b)
for file in files:
    with open(b + file, 'r', encoding='UTF8') as f:
        json_data = json.load(f)
    
    img_width = json_data['images'][0]['width']
    img_height = json_data['images'][0]['height']
    bbox = json_data['annotations'][0]['bbox']
    
    base_name = os.path.splitext(file)[0]
    a = 'images/1/' + base_name + '.jpg'
    img = cv2.imread(a, cv2.IMREAD_COLOR)
    
    
    cv2.putText(img, f'class: 0', (int(bbox[0][0]), int(bbox[0][1])),\
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,50,255), 2)
    cv2.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])),\
                (int(bbox[1][0]), int(bbox[1][1])), (255,50,255), 2)
    x_center = int((bbox[0][0] + bbox[1][0]) / 2)
    y_center = int((bbox[0][1] + bbox[1][1]) / 2)
    width = int((bbox[1][0] - bbox[0][0]))
    height = int((bbox[1][1] - bbox[0][1]))
    cv2.circle(img, (x_center,y_center), 10, color=(255,50,0), thickness=-1, lineType=None,shift=None)
    cv2.line(img, (int(bbox[0][0]) , int(bbox[1][1])),(int(bbox[1][0]) , int(bbox[1][1])), color=(255,50,0), thickness=5, lineType=None, shift=None)
    cv2.line(img, (int(bbox[1][0]) , int(bbox[0][1])),(int(bbox[1][0]) , int(bbox[1][1])), color=(255,50,0), thickness=5, lineType=None, shift=None)

    cv2.imshow('test', img)
    cv2.waitKey(1)
    cv2.imwrite('imge_test'+a, img)
    
    
    txt_path = os.path.join('labels', base_name + '.txt')
    save_to_yolo_format(0, bbox, img_width, img_height, txt_path)

cv2.destroyAllWindows()
