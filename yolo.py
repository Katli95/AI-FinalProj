import os
import sys
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.backend import print_tensor

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

CLASSES = ["sphere", "can", "bottle"]
CLASS_WEIGHTS = np.array([1.5,0.7,0.5], dtype='float32')

from dataHandler import BatchGenerator, read_Imgs
import cv2

from utils import decode_netout, compute_overlap, compute_ap, normalizeImage, draw_boxes

# Constants
NUM_CLASSES = len(CLASSES)
BOUNDING_BOX_ATTRIBUTES = 5  # 4 for x,y,w,h and 1 for confidence

GRID_DIM = 7
NUM_BOXES = 2
NUM_DENSE_NODES = (GRID_DIM*GRID_DIM*NUM_BOXES*(BOUNDING_BOX_ATTRIBUTES+NUM_CLASSES))
OUTPUT_SHAPE = (GRID_DIM, GRID_DIM, NUM_BOXES, BOUNDING_BOX_ATTRIBUTES+NUM_CLASSES)

BATCH_SIZE = 8
INPUT_SIZE = 448

WEIGHT_PATH = "tiny_yolov1_weights.h5"

NO_OBJECT_SCALE = .5
OBJECT_SCALE = 1.0
COORD_SCALE = 5.0
CLASS_SCALE = 1.0


class YOLO(object):
    def __init__(self):

        self.input_size = INPUT_SIZE
        self.batch_size = BATCH_SIZE

        self.labels = CLASSES
        self.nb_class = len(CLASSES)
        self.nb_box = NUM_BOXES

        self.class_wt = np.ones(self.nb_class, dtype='float32')

        ##########################
        # Make the model
        ##########################
        self.model = self.getTinyYoloNetwork()

        self.model.summary()

    def getTinyYoloNetwork(self):

        model = Sequential()

        # Layer 1
        model.add(Conv2D(16, (3, 3), strides=(1, 1), name='conv_1',
            input_shape=(self.input_size, self.input_size, 3),padding='same', use_bias=False,))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(name='norm_1'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 2
        model.add(Conv2D(32, (3,3), strides=(1,1), name="conv_2", padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(name='norm_2'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 3
        model.add(Conv2D(64, (3,3), strides=(1,1), name="conv_3", padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(name='norm_3'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 4
        model.add(Conv2D(128, (3,3), strides=(1,1), name="conv_4", padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(name='norm_4'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 5
        model.add(Conv2D(256, (3,3), strides=(1,1), name="conv_5", padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(name='norm_5'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 6
        model.add(Conv2D(512, (3,3), strides=(1,1), name="conv_6", padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(name='norm_6'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Layer 7
        model.add(Conv2D(1024, (3,3), strides=(1,1), name="conv_7", padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(name='norm_7'))

        # Layer 8
        model.add(Conv2D(256, (3,3), strides=(1,1), name="conv_8", padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization(name='norm_8'))

        # Layer 9
        model.add(Flatten())
        model.add(Dense(NUM_DENSE_NODES, name='dense_1'))

        model.add(Reshape(OUTPUT_SHAPE))

        if os.path.isfile(WEIGHT_PATH):
            print("Reloading weights from " + WEIGHT_PATH)
            model.load_weights(WEIGHT_PATH)

        return model

    def predict(self, image):
        height, width, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))

        image = normalizeImage(image)

        # input_image = image[:, :, ::-1]  # Reverses the channels
        input_image = np.expand_dims(image, 0)

        netout = self.model.predict([input_image])[0]
        boxes = decode_netout(netout, self.nb_class)

        return boxes

    def evaluate(self,
                 generator,
                 iou_threshold=0.3,
                 save_path=None):

        all_detections = [[None for i in range(
            NUM_CLASSES)] for j in range(generator.size())]
        all_annotations = [[None for i in range(
            NUM_CLASSES)] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes = self.predict(raw_image)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax *
                                        raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

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
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(
                        np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / \
                np.maximum(true_positives + false_positives,
                           np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

    def train(self,
              train_times,    # the number of time to repeat the training set, often used for small datasets
              valid_times,    # the number of times to repeat the validation set, often used for small datasets
              nb_epochs,      # number of epoches
              learning_rate,  # the learning rate
              warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
              debug=False):

        self.batch_size = BATCH_SIZE
        self.object_scale = OBJECT_SCALE
        self.no_object_scale = NO_OBJECT_SCALE
        self.coord_scale = COORD_SCALE
        self.class_scale = CLASS_SCALE
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
            'IMAGE_H': self.input_size,
            'IMAGE_W': self.input_size,
            'GRID_H': GRID_DIM,
            'GRID_W': GRID_DIM,
            'BOX': self.nb_box,
            'LABELS': self.labels,
            'CLASS': self.nb_class,
            'BATCH_SIZE': self.batch_size,
        }

        train_generator = BatchGenerator(train_imgs,
                                         generator_config)
        test_generator = BatchGenerator(train_imgs,
                                         generator_config)
        # TODO: FIX
        # test_generator = BatchGenerator(test_imgs,
        #                                 generator_config)

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9,
                         beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   patience=3,
                                   mode='min',
                                   verbose=1)
        checkpoint = ModelCheckpoint(WEIGHT_PATH,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min',
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                                  histogram_freq=0,
                                  # write_batch_performance=True,
                                  write_graph=True,
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=len(
                                     train_generator) * train_times,
                                 epochs=warmup_epochs + nb_epochs,
                                 verbose=2 if debug else 1,
                                 validation_data=test_generator,
                                 validation_steps=len(
                                     test_generator) * valid_times,
                                 callbacks=[#early_stop,
                                            checkpoint, tensorboard],
                                 workers=1,
                                 max_queue_size=3)

        ############################################
        # Compute mAP on the validation set
        ############################################
        average_precisions = self.evaluate(test_generator)

        # print evaluation
        for label, average_precision in average_precisions.items():
            print(self.labels[label], '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(
            sum(average_precisions.values()) / len(average_precisions)))

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4] #Masking out the four dimension Batch, width, height, num_box

        # cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        # cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        # cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])

        """
        Load prediction
        """
        ### get x and y in terms of grid
        pred_box_xy = y_pred[..., :2] #+ cell_grid
                
        ### account for network predicting squares of w and h
        sqrt_pred_box_wh = y_pred[..., 2:4]
        pred_box_wh = tf.square(sqrt_pred_box_wh)    
            
        ### confidence should be in [0,1]
        pred_box_conf = y_pred[..., 4]
                
        ### class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Load ground truth
        """
        ### x and y center of boxes
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically

        ### Find iou for conf given obj, else 0
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.where(tf.less(tf.abs(union_areas), 1e-4), union_areas, tf.truediv(intersect_areas, union_areas)) 
        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = y_true[..., 5:]

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        obj_mask_for_mult_attr = tf.expand_dims(y_true[..., 4], axis=-1)
        obj_mask_for_one_attr = y_true[..., 4]
        coord_mask =  obj_mask_for_mult_attr * COORD_SCALE
        conf_mask_obj = obj_mask_for_one_attr * OBJECT_SCALE
        conf_mask_no_obj = (1-obj_mask_for_one_attr) * NO_OBJECT_SCALE

        """
        Summarize the loss
        """
        # nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        # nb_conf_box_neg = tf.reduce_sum(tf.to_float(conf_mask_no_obj > 0.0))
        # nb_conf_box_pos = tf.reduce_sum(tf.to_float(conf_mask_obj > 0.0))
        # nb_obj = tf.reduce_sum(tf.to_float(obj_mask_for_one_attr > 0.0))

        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask, axis=[1,2,3,4]) #/ (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask, axis=[1,2,3,4]) #/ (nb_coord_box + 1e-6) / 2.
        loss_conf_neg = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask_no_obj, axis=[1,2,3]) #/ (nb_conf_box_neg + 1e-6) / 2.
        loss_conf_pos = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask_obj, axis=[1,2,3]) #/ (nb_conf_box_pos + 1e-6) / 2.
        loss_class = tf.reduce_sum(tf.square(true_box_class - pred_box_class)* obj_mask_for_mult_attr, axis=[1,2,3,4])#/(nb_obj + 1e-6)

        loss = loss_xy + loss_wh + loss_conf_pos + loss_conf_neg + loss_class

        # zero_losses = [tf.less(x,1e-5).eval() for x in [loss_xy, loss_wh, loss_conf_neg, loss_conf_pos, loss_class]]
        
        # loss_names = ["loss_xy", "loss_wh", "loss_conf_neg", "loss_conf_pos", "loss_class"]

        import keras.backend as K

        if not y_true.op.type == 'Placeholder':

            file_path = "./debug/0_loss.npz"
            if(os.path.isfile(file_path)): 
                os.remove(file_path)
            with open(file_path, "wb+") as file:
                np.savez(file, true=K.tf.round(y_true), pred=K.tf.round(y_pred))
                # np.savez(file, true=y_true.eval(session=tf.keras.backend.get_session()), pred=y_pred.eval(session=tf.keras.backend.get_session()))
            print("Model saved in path: %s" % file_path)

        if self.debug:
                total_recall = tf.Variable(0.)
                nb_true_box = tf.reduce_sum(y_true[..., 4])
                nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
                
                current_recall = nb_pred_box/(nb_true_box + 1e-6)
                total_recall = tf.assign_add(total_recall, current_recall) 

                loss = tf.Print(loss, [loss_xy], message='Loss XY \t')
                loss = tf.Print(loss, [loss_wh], message='Loss WH \t')
                loss = tf.Print(loss, [loss_conf_pos], message='Loss Conf Pos \t')
                loss = tf.Print(loss, [loss_conf_neg], message='Loss Conf Neg\t')
                loss = tf.Print(loss, [loss_class], message='Loss Class \t')
                loss = tf.Print(loss, [loss], message='Total Loss \t')
                loss = tf.Print(loss, [current_recall], message='Current Recall \t')
                loss = tf.Print(loss, [total_recall], message='Average Recall \t')

        return loss

    def testImg(self, imgPath):
        img = cv2.imread(imgPath)
        boxes = self.predict(img)
        img = draw_boxes(img, boxes, CLASSES)
        cv2.imwrite("./data/output/" + imgPath[:-4].split("/")[-1] + "_detected" + imgPath[-4:], img)

