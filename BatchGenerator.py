import BoundBox
import numpy as np
import cv2 #open cv


class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)    

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))           # input images TODO: Gerir þetta ráð fyrir að allar myndir séu jafn stórar?
        b_batch = np.zeros((r_bound - l_bound, 1 , 1, 1, self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size

            #Þessi kóði var gerður inn í fallinu aug_image og skilaði all_objs og img 
            image_name = train_instance['filename']
            img = cv2.imread(image_name)
            obj = train_instance['object']
            
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                center_x = .5*(obj['xmin'] + obj['xmax'])
                center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                center_y = .5*(obj['ymin'] + obj['ymax'])
                center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                    obj_indx  = self.config['LABELS'].index(obj['name'])
                    
                    center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                    center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                    
                    box = [center_x, center_y, center_w, center_h]

                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou     = -1
                    
                    shifted_box = BoundBox(0, 
                                            0,
                                            center_w,                                                
                                            center_h)
                    
                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou    = bbox_iou(shifted_box, anchor)
                        
                        if max_iou < iou:
                            best_anchor = i
                            max_iou     = iou
                            
                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                    y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1
                    
                    # assign the true box to b_batch
                    b_batch[instance_count, 0, 0, 0, true_box_index] = box
                    
                    true_box_index += 1
                    true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                        
            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12), 
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        
                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1  

        #print(' new batch created', idx)

        return [x_batch, b_batch], y_batch