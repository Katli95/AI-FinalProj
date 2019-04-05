from dataHandler import *
from utils import decode_netout
from config import CLASSES

imgs = read_Imgs()

train_instance = {
    'objects':
    [
        {'name': 'sphere', 'xmin': 1358, 'ymin': 107, 'xmax': 1472, 'ymax': 226},
        {'name': 'can', 'xmin': 1215, 'ymin': 533, 'xmax': 1412, 'ymax': 616},
        {'name': 'bottle', 'xmin': 555, 'ymin': 548, 'xmax': 739, 'ymax': 736},
        {'name': 'bottle', 'xmin': 1232, 'ymin': 419, 'xmax': 1475, 'ymax': 505}
    ],
    'filename': './data/img/685be22c-c384-4302-b0f9-080b77f10a27.jpg',
    'width': 1920,
    'height': 1080}

generator_config = {
    'IMAGE_H': 448,
    'IMAGE_W': 448,
    'GRID_H': 7,
    'GRID_W': 7,
    'BOX': 2,
    'LABELS': ['sphere', 'can', 'bottle'],
    'CLASS': 3,
    'BATCH_SIZE': 1,
}

bg = BatchGenerator(imgs, generator_config)
raw_box = bg.parse_objects(train_instance['objects'])
print(raw_box)
bg.encode_boxes_for_netout([raw_box], train_instance)


def printGround():
    true_netout = np.load("./debug/man_true_batch.npy")[0]
    boxes = decode_netout(true_netout, 3)
    img = cv2.imread("./data/img/022dbb19-2f9c-4fea-bfd1-292260878db0.jpg")
    img = draw_boxes(img, boxes, ['sphere', 'can', 'bottle'])
    cv2.imwrite("./man.jpg", img)
