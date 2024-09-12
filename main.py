import argparse
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import time
from imutils.video import FPS

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-p', '--prototxt', required=True, help='path to prototxt file')
    parse.add_argument('-m', '--model', required=True, help='path to the Caffe pre-trained model')
    parse.add_argument('-c', '--confidence', type=float, default=0.2, help='minimum probability to filter weak detections')
    args = vars(parse.parse_args())
    return args

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def load_model(prototxt, model):
    print("Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    return net

def start_video_stream():
    print("Starting video stream...")
    video_stream = VideoStream(src=0).start()
    time.sleep(2.0)
    return video_stream

if __name__ == '__main__':
    args = parse_args()
    net = load_model(args['prototxt'], args['model'])
    video_stream = start_video_stream()
    fps = FPS().start()

    while True:
        frame = video_stream.read()
        frame = imutils.resize(frame, width=400)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            scalefactor=0.007843, # 1/127.5
            size=(300, 300), 
            mean=127.5 # usually values [0, 225] but now [-127.5, 127.5]
        ) # This normalizes image pixels to range [-1, 1]

        net.setInput(blob)
        detections = net.forward()


