import os
import copy
import random
import xml.etree.ElementTree as ET

import cv2 #open cv
import numpy as np
from keras.utils import Sequence

from utils import BoundBox, normalizeImage
from config import CLASSES
import data_aug.data_aug as aug

imgDir = "./data/img"
annDir = "./data/annotations"

def read_Imgs():
    all_imgs = []

    for annotationFile in sorted(os.listdir(annDir)):
        img = {'objects': [], 'filename': imgDir + "/" + annotationFile.replace(".xml", ".jpg")}

        tree = ET.parse(annDir + "/" + annotationFile)

        for elem in tree.iter():
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text.lower()
                        img['objects'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        # TODO: REMOVE 'and all(...' ONLY FOR TESTING THE NETWORK ON SPHERES!!  
        if len(img['objects']) > 0:
            img['objects'].sort(key=lambda x: CLASSES.index(x['name']))
            all_imgs += [img]

    return all_imgs

class BatchGenerator(Sequence):
    def __init__(self, images, config, should_aug=True, checkSanity = False):
        self.generator = None

        self.images = images
        np.random.shuffle(self.images)
        
        self.config = config
        self.should_aug = should_aug
        self.checkSanity = checkSanity
        self.seq = aug.Sequence([aug.RandomRotate(10), aug.RandomHSV(40, 40, 30),aug.RandomHorizontalFlip(), aug.RandomScale(), aug.RandomTranslate(0.2,diff=True)])

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def size(self):
        return len(self.images)    

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['objects']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.resize(cv2.imread(self.images[i]['filename']), (self.config['IMAGE_W'],self.config['IMAGE_H']))

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))           # input images TODO: Gerir þetta ráð fyrir að allar myndir séu jafn stórar?
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size

            #Þessi kóði var gerður inn í fallinu aug_image og skilaði all_objs og img 
            image_name = train_instance['filename']
            img = cv2.imread(image_name)
            all_objs = copy.deepcopy(train_instance['objects'])
            
            boxes = self.parse_objects(all_objs)

            if self.should_aug and random.random() <= 1.0:
                try:
                    img, boxes = self.seq(img.copy(), boxes.copy())
                except:
                    image_name = train_instance['filename']
                    img = cv2.imread(image_name)
                    all_objs = copy.deepcopy(train_instance['objects'])
                    
                    boxes = self.parse_objects(all_objs)

            img = cv2.resize(img, (self.config['IMAGE_W'],self.config['IMAGE_H']))

            boxes = self.encode_boxes_for_netout(boxes, train_instance)

            for final_box in boxes:
                box = final_box[0]
                grid_x, grid_y, obj_indx = np.array(final_box[1]).astype(int)
            
                nextBoxIndex = 0
                if y_batch[instance_count, grid_y, grid_x, 0, 4] == 0:
                    nextBoxIndex = 0
                elif y_batch[instance_count, grid_y, grid_x, 1, 4] == 0:
                    nextBoxIndex = 1
                else:
                    continue

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                y_batch[instance_count, grid_y, grid_x, nextBoxIndex, 0:4] = box
                y_batch[instance_count, grid_y, grid_x, nextBoxIndex, 4  ] = 1.
                y_batch[instance_count, grid_y, grid_x, nextBoxIndex, 5+obj_indx] = 1.
                nextBoxIndex+=1

            # assign input image to x_batch
            if self.checkSanity:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img, obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12), 
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        
                x_batch[instance_count] = normalizeImage(img)
            else:
                x_batch[instance_count] = normalizeImage(img)

            # increase instance counter in current batch
            instance_count += 1  

        #print(' new batch created', idx)
        np.save("./debug/auto_true_batch", y_batch)
        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.images)

    def parse_objects(self, objects):
        boxes = []
        for obj in objects:
            if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                obj_indx  = self.config['LABELS'].index(obj['name'])
                boxes += [[obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj_indx]]
        return np.array(boxes, dtype='float64')

    def encode_boxes_for_netout(self, raw_boxes, train_instance):
        translated_boxes = []
        for box in raw_boxes:
            center_x = (box[0] + box[2])/2
            center_x_relative_to_image = center_x / train_instance['width']
            center_x_in_box_units = center_x_relative_to_image * self.config['GRID_W']

            center_y = (box[1] + box[3])/2
            center_y_relative_to_image = center_y / train_instance['height']
            center_y_in_box_units = center_y_relative_to_image * self.config['GRID_H']

            grid_x = int(np.floor(center_x_in_box_units))
            grid_y = int(np.floor(center_y_in_box_units))
            center_x_rel_to_box = center_x_in_box_units % 1.
            center_y_rel_to_box = center_y_in_box_units % 1.
            
            if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                obj_indx  = int(box[-1])
                
                center_w = np.sqrt((box[2] - box[0]) / float(train_instance['width'])) # relative to image
                center_h = np.sqrt((box[3] - box[1]) / float(train_instance['height'])) # relative to image
                
                translated_boxes += [[[center_x_rel_to_box, center_y_rel_to_box, center_w, center_h], [grid_x, grid_y, obj_indx]]]

        return translated_boxes
