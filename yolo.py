import os
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from BatchGenerator import BatchGenerator 
import cv2

from utils import decode_netout, compute_overlap, compute_ap

from dataHandler import read_Imgs

# Constants
CLASSES = ["sphere", "can", "bottle"]
NUM_CLASSES = len(CLASSES)
BoundingBoxOverhead = 5 # 4 for x,y,w,h and 1 for confidence

GRID_W = GRID_H = 13
NB_BOXES = 2
BATCH_SIZE = 16
INPUT_SIZE = 416
MAXIMUM_NUMBER_OF_BOXES_PER_IMAGE = 10
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
           5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

WEIGHT_PATH = "tiny_yolo_weights.h5"

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0

class YOLO(object):
    def __init__(self):

        self.input_size = INPUT_SIZE
        self.batch_size = BATCH_SIZE
        self.max_box_per_image = MAXIMUM_NUMBER_OF_BOXES_PER_IMAGE
        self.anchors = ANCHORS

        self.labels = CLASSES
        self.nb_class = len(CLASSES)
        self.nb_box = len(self.anchors)//2
        
        self.class_wt = np.ones(self.nb_class, dtype='float32')


        ##########################
        # Make the model
        ##########################
        self.model = self.getTinyYoloNetwork()

        self.model.summary()

    def getTinyYoloNetwork(self):
        self.true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))

        input_layer = Input(shape=(self.input_size, self.input_size, 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_layer)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0,2):
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        # x = Model(input_layer, x)

        # print(x.get_output_shape_at(-1)[1:3])
        # x = x(Input(shape=(self.input_size, self.input_size, 3)))

        # Layer 9
        x = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(x)

        x = Reshape((GRID_H, GRID_W, self.nb_box, 4 + 1 + self.nb_class))(x)

        #This is done to incorporate the true boxes as a second parameter into the network while training
        x = Lambda(lambda args: args[0])([x, self.true_boxes])

        model = Model([input_layer, self.true_boxes], x)

        # model = Model(input_layer, x)
        
        if os.path.isfile(WEIGHT_PATH):
            model.load_weights(WEIGHT_PATH)

        # initialize the weights of the detection layer
        # layer = model.layers[-4]
        # weights = layer.get_weights()

        # new_kernel = np.random.normal(
        #     size=weights[0].shape)/(GRID_H*GRID_W)
        # new_bias = np.random.normal(
        #     size=weights[1].shape)/(GRID_H*GRID_W)

        # layer.set_weights([new_kernel, new_bias])

        return model

    def predict(self, image):
        height, width, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))

        image = self.normalizeImage(image)

        input_image = image[:,:,::-1] #Reverses the channels
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = decode_netout(netout, self.anchors, self.nb_class)

        return boxes

    def evaluate(self,
                 generator,
                 iou_threshold=0.3,
                 save_path=None):

        all_detections  = [[None for i in range(NUM_CLASSES)] for j in range(generator.size())]
        all_annotations = [[None for i in range(NUM_CLASSES)] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self.predict(raw_image)

            
            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])        
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(NUM_CLASSES):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(NUM_CLASSES):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        average_precisions = {}

        for label in range(NUM_CLASSES):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                            false_positives = np.append(false_positives, 1)
                            true_positives  = np.append(true_positives, 0)
                            continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        return average_precisions

    def train(self,
                train_times,    # the number of time to repeat the training set, often used for small datasets
                valid_times,    # the number of times to repeat the validation set, often used for small datasets
                nb_epochs,      # number of epoches
                learning_rate,  # the learning rate
                batch_size,     # the size of the batch
                warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                saved_weights_name=WEIGHT_PATH,
                debug=False):     
        
        self.batch_size = batch_size
        self.object_scale    = OBJECT_SCALE
        self.no_object_scale = NO_OBJECT_SCALE
        self.coord_scale     = COORD_SCALE
        self.class_scale     = CLASS_SCALE
        self.debug = debug

        # Load Images
        train_imgs = read_Imgs()

        validStartIndex = int(len(train_imgs)*0.8)
        test_imgs = train_imgs[validStartIndex:]
        train_imgs = train_imgs[:validStartIndex]
        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_size, 
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : GRID_H,
            'GRID_W'          : GRID_W,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : self.nb_class,
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }    

        train_generator = BatchGenerator(train_imgs, 
                                    generator_config, 
                                    norm=self.normalizeImage)

        test_generator = BatchGenerator(test_imgs, 
                                    generator_config, 
                                    norm=self.normalizeImage,
                                    jitter=False)

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss', 
                        min_delta=0.001, 
                        patience=3, 
                        mode='min', 
                        verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name, 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='min', 
                                    period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'), 
                                histogram_freq=0, 
                                #write_batch_performance=True,
                                write_graph=True, 
                                write_images=False)

        ############################################
        # Start the training process
        ############################################        

        self.model.fit_generator(generator        = train_generator, 
                                steps_per_epoch  = len(train_generator) * train_times, 
                                epochs           = warmup_epochs + nb_epochs, 
                                verbose          = 2 if debug else 1,
                                validation_data  = test_generator,
                                validation_steps = len(test_generator) * valid_times,
                                callbacks        = [early_stop, checkpoint, tensorboard], 
                                workers          = 3,
                                max_queue_size   = 8)      

        ############################################
        # Compute mAP on the validation set
        ############################################
        average_precisions = self.evaluate(test_generator)     

        # print evaluation
        for label, average_precision in average_precisions.items():
            print(self.labels[label], '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [
                             GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(
            tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        # adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        # adjust w and h
        pred_box_wh = tf.exp(
            y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        # adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        # adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        # adjust x and y
        # relative position to the containing cell
        true_box_xy = y_true[..., 0:2]

        # adjust w and h
        # number of cells accross, horizontally and vertically
        true_box_wh = y_true[..., 2:4]

        # adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        # adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        # coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

        # confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + \
            tf.to_float(best_ious < 0.6) * \
            (1 - y_true[..., 4]) * NO_OBJECT_SCALE

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

        # class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * \
            tf.gather(self.class_wt, true_box_class) * CLASS_SCALE

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = [true_box_xy,true_box_wh,coord_mask]

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(
            tf.square(true_box_xy-pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(
            tf.square(true_box_wh-pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(
            tf.square(true_box_conf-pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(
            loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(
                true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

            current_recall = nb_pred_box/(nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_xy],
                            message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh],
                            message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf],
                            message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class],
                            message='Loss Class \t', summarize=1000)
            loss = tf.Print(
                loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall],
                            message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall/seen],
                            message='Average Recall \t', summarize=1000)

        return loss

    def normalizeImage(self, image):
        return image / INPUT_SIZE