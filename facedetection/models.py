import cv2.cv2 as cv2
import tensorflow as tf
import dlib
import numpy as np
import time

from mtcnn.mtcnn import MTCNN
from moviepy.editor import *


class OpencvHaarCascade:
    def __init__(self):
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect(self, frame):
        bboxes = self.detector.detectMultiScale(frame, 1.5, 5)
        return bboxes


class DlibHOG:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        dets, scores, idx = self.detector.run(image)
        bboxes = []
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = int(d.left()), int(d.top()), int(d.right()), int(d.bottom())
            bboxes.append([x1, y1, x2, y2])
        return np.array(bboxes)


class TensorFlowMTCNN:
    def __init__(self):
        self.detector = MTCNN()

    def detect(self, frame):

        faces = self.detector.detect_faces(frame)
        bboxes = [face['box'] for face in faces]
        return bboxes


class TensorFlowMobileNetSSD:
    def __init__(self, det_threshold=0.3, model_path='model/frozen_inference_graph_face.pb'):
        self.det_threshold = det_threshold
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            # od_graph_def = tf.GraphDef()
            od_graph_def = tf.compat.v1.GraphDef()
            # with tf.gfile.GFile(model_path, 'rb') as fid:
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            # config = tf.ConfigProto()
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            # self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)

    def detect(self, image):
        h, w, c = image.shape

        image_np = image
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        filtered_score_index = np.argwhere(scores >= self.det_threshold).flatten()
        selected_boxes = boxes[filtered_score_index]

        faces = np.array([[int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h),] for y1, x1, y2, x2 in selected_boxes])

        return faces


def run_test():
    pass


if __name__ == '__main__':

    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out_name = f'record_face_detection_{round(time.time(),3)}.avi'
    # out = cv2.VideoWriter(out_name, fourcc, 20.0, (640, 480))

    detector = OpencvHaarCascade()  # TODO
    detector = TensorFlowMTCNN()
    detector = DlibHOG()
    detector = TensorFlowMobileNetSSD()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t0 = time.time()
            frame = cv2.flip(frame, 1)
            cropped = frame[206-100:206+100, 256-100:256+100]

            bboxes = detector.detect(cropped)

            for box in bboxes:
                x, y, width, height = box
                x, y = x+256-100, y+206-100
                x2, y2 = x+width, y+height
                cv2.rectangle(frame, (x,y), (x2,y2), (0,100,100), 5)

            cv2.ellipse(frame, (256,206), (100,100), 0, 0, 360, 50, 5)
            cv2.rectangle(frame, (256-100, 206-100), (256+100, 206+100), (0,0,255), 10)
            t1 = time.time()
            font = cv2.FONT_HERSHEY_PLAIN
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)

            thickness = 2
            message = f'{round(t1-t0, 3)} sec per frame'

            cv2.putText(frame, message, org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("preview", frame)
            # out.write(frame)
            key = cv2.waitKey(20)

            if key == 27:  # exit on ESC
                break
        else:
            break

    cv2.destroyWindow("Preview")
    cap.release()
    # out.release()
