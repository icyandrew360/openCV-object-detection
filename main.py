import argparse
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import time
from imutils.video import FPS
from enum import Enum

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-p', '--prototxt', required=True, help='path to prototxt file')
    parse.add_argument('-m', '--model', required=True, help='path to the Caffe pre-trained model')
    parse.add_argument('-c', '--confidence', type=float, default=0.6, help='minimum probability to filter weak detections')
    args = vars(parse.parse_args())
    return args

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

Prototxts = {
    'MobileNetSSD': 'MobileNetSSD.txt',
}
Models = {
    'MobileNetSSD': 'MobileNetSSD_deploy.caffemodel',
}

def load_model(prototxt, model):
    print("Loading model...")
    prototxt = Prototxts[prototxt]
    model = Models[model]
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

        for i in np.arange(0, detections.shape[2]): #detections.shape[2] is the number of detections
            confidence = detections[0, 0, i, 2] # accesses the confidence score of the i-th detection

            if confidence > args['confidence']:
                detection_index = int(detections[0, 0, i, 1]) # accesses the index of the class label
                bounding_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # bounding box coordinates relative to the image size
                # Multiplying by the image size scales the bounding box to the image size
                (startX, startY, endX, endY) = bounding_box.astype('int')

                label = "{}: {:.2f}%".format(CLASSES[detection_index], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[detection_index], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    frame,
                    label, 
                    (startX, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    COLORS[detection_index], 
                    2
                )

            
        cv2.imshow('Frame', frame)
            
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        fps.update()

    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

