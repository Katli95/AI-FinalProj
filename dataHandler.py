import os
import copy
import xml.etree.ElementTree as ET

import cv2 #open cv
import numpy as np
from keras.utils import Sequence

from utils import BoundBox, normalizeImage
from yolo import CLASSES

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
        if len(img['objects']) > 0 and all(x['name'] == 'sphere' for x in img['objects']):
            img['objects'].sort(key=lambda x: CLASSES.index(x['name']))
            all_imgs += [img]

    return all_imgs

class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config,
                       checkSanity = False):
        self.generator = None

        self.images = images
        np.random.shuffle(self.images)
        
        self.config = config
        self.checkSanity = checkSanity

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
            img = cv2.resize(cv2.imread(image_name), (self.config['IMAGE_W'],self.config['IMAGE_H']))
            all_objs = copy.deepcopy(train_instance['objects'])
            
            nextBoxIndex = 0
            # construct output from object's x, y, w, h
            for obj in all_objs:
                if nextBoxIndex >= 2:
                    break
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    center_x = (obj['xmin'] + obj['xmax'])/2
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = (obj['ymin'] + obj['ymax'])/2
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])
                        
                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                        
                        box = [center_x, center_y, center_w, center_h]
                        if y_batch[instance_count, grid_y, grid_x, nextBoxIndex, 4] == 1:
                            nextBoxIndex+=1

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, nextBoxIndex, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, nextBoxIndex, 4  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, nextBoxIndex, 5+obj_indx] = 1
                            
            # assign input image to x_batch
            if self.checkSanity:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12), 
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        
                x_batch[instance_count] = img
            else:
                x_batch[instance_count] = normalizeImage(img)

            # increase instance counter in current batch
            instance_count += 1  

        #print(' new batch created', idx)

        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.images)