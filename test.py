from yolo import YOLO
import cv2

test = YOLO()
#test.train(2,2,30,10**-3,16,0)
test.predict(cv2.imread("./data/img/0cec79be-790b-48f0-a23c-eef4f876d1bc.jpg"))