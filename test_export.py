import cv2
import copy
import numpy as np

CLASSES = ['sphere', 'can', 'bot']

def get_true(train_instance):
    image_name = train_instance['filename']
    img = cv2.resize(cv2.imread(image_name), (448,448))
    all_objs = copy.deepcopy(train_instance['objects'])

    y_batch = np.zeros((8,7,7,2,8))
    
    nextBoxIndex = 0
    # construct output from object's x, y, w, h
    for obj in all_objs:
        if nextBoxIndex >= 2:
            break
        if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
            center_x = (obj['xmin'] + obj['xmax'])/2
            center_x = center_x / (float(448) / 448)
            center_y = (obj['ymin'] + obj['ymax'])/2
            center_y = center_y / (float(448) / 448)

            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            center_x = center_x % 1.
            center_y = center_y % 1.

            if grid_x < 448 and grid_y < 448:
                obj_indx  = CLASSES.index(obj['name'])
                
                center_w = (obj['xmax'] - obj['xmin']) / (float(448) / 448) # unit: grid cell
                center_h = (obj['ymax'] - obj['ymin']) / (float(448) / 448) # unit: grid cell
                
                box = [center_x, center_y, center_w, center_h]
                if y_batch[:, grid_y, grid_x, nextBoxIndex, 4] == 1:
                    nextBoxIndex+=1

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                y_batch[:, grid_y, grid_x, nextBoxIndex, 0:4] = box
                y_batch[:, grid_y, grid_x, nextBoxIndex, 4  ] = 1.
                y_batch[:, grid_y, grid_x, nextBoxIndex, 5+obj_indx] = 1.
    return y_batch
