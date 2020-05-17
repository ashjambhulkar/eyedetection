import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        loadUi('frontend/home.ui', self)
        with open("frontend/style.css", "r") as css:
            self.setStyleSheet(css.read())
        self.face_decector, self.eye_detector, self.detector = self.cv_variables()
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.camera_on = False
        self.previous_right_keypoints = None
        self.previous_left_keypoints = None
        self.previous_right_blob_area = None
        self.previous_left_blob_area = None
    
    def cv_variables(self):
        face_detector = cv2.CascadeClassifier(
            os.path.join("Classifiers", "haar", "haarcascade_frontalface_default.xml"))
        eye_detector = cv2.CascadeClassifier(os.path.join(
            "Classifiers", "haar", 'haarcascade_eye.xml'))
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1500sub
        detector = cv2.SimpleBlobDetector_create(detector_params)

        return face_detector, eye_detector, detector

    def detect_face(self, img, img_gray, cascade):
        coords = cascade.detectMultiScale(img, 1.3, 5)
        if len(coords) > 1:
            biggest = (0, 0, 0, 0)
            for i in coords:
                if i[3] > biggest[3]:
                    biggest = i
            biggest = np.array([i], np.int32)
        elif len(coords) == 1:
            biggest = coords
        else:
            return None, None, None, None, None, None
        for (x, y, w, h) in biggest:
            frame = img[y:y + h, x:x + w]
            frame_gray = img_gray[y:y + h, x:x + w]
            lest = (int(w * 0.1), int(w * 0.45))
            rest = (int(w * 0.55), int(w * 0.9))
            X = x
            Y = y

        return frame, frame_gray, lest, rest, X, Y

    def detect_eyes(self, img, img_gray, lest, rest, cascade):
        left_eye = None
        right_eye = None
        left_eye_gray = None
        right_eye_gray = None
        coords = cascade.detectMultiScale(img_gray, 1.3, 5)

        if coords is None or len(coords) == 0:
            pass
        else:
            for (x, y, w, h) in coords:
                eyecenter = int(float(x) + (float(w) / float(2)))
                if lest[0] < eyecenter and eyecenter < lest[1]:
                    left_eye = img[y:y + h, x:x + w]
                    left_eye_gray = img_gray[y:y + h, x:x + w]
                elif rest[0] < eyecenter and eyecenter < rest[1]:
                    right_eye = img[y:y + h, x:x + w]
                    right_eye_gray = img_gray[y:y + h, x:x + w]
                else:
                    pass
        return left_eye, right_eye, left_eye_gray, right_eye_gray
    

    def process_eye(self, img, threshold, detector, prevArea=None):
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=4)
        img = cv2.medianBlur(img, 5)
        keypoints = detector.detect(img)
        if keypoints and prevArea and len(keypoints) > 1:
            tmp = 1000
            for keypoint in keypoints:  # filter out odd blobs
                if abs(keypoint.size - prevArea) < tmp:
                    ans = keypoint
                    tmp = abs(keypoint.size - prevArea)
            keypoints = np.array(ans)

        return keypoints

    def draw_blobs(self, img, keypoints):
        cv2.drawKeypoints(img, keypoints, img, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def start_webcam(self):
        if not self.camera_on:
            self.capture = cv2.VideoCapture(0)
            self.camera_on = True
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(1)

    def stop_webcam(self):
        if self.camera_on:
            self.capture.release()
            self.timer.stop()
            self.camera_on = False

    def update_frame(self):

        _, base_image = self.capture.read()
        self.display_image(base_image)

        processed_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)

        face_area, face_area_gray, left_eye_position, right_eye_position, _, _ = self.detect_face(
            base_image, processed_image, self.face_decector)

        if face_area is not None:
            left_eye_area, right_eye_area, left_eye_area_gray, right_eye_area_gray = self.detect_eyes(face_area, face_area_gray, left_eye_position,
            right_eye_position, self.eye_detector)

            if right_eye_area is not None:
                if self.rightEyeCheckbox.isChecked():
                    right_eye_threshold = self.rightEyeThreshold.value()
                    right_keypoints, self.previous_right_keypoints, self.previous_right_blob_area = self.get_keypoints(
                        right_eye_area, right_eye_area_gray, right_eye_threshold,
                        previous_area=self.previous_right_blob_area,
                        previous_keypoint=self.previous_right_keypoints)
                    self.draw_blobs(right_eye_area, right_keypoints)

                right_eye_area = np.require(right_eye_area, np.uint8, 'C')
                self.display_image(right_eye_area, window='right')

            if left_eye_area is not None:
                if self.leftEyeCheckbox.isChecked():
                    left_eye_threshold = self.leftEyeThreshold.value()
                    left_keypoints, self.previous_left_keypoints, self.previous_left_blob_area = self.get_keypoints(
                        left_eye_area, left_eye_area_gray, left_eye_threshold,
                        previous_area=self.previous_left_blob_area,
                        previous_keypoint=self.previous_left_keypoints)
                    self.draw_blobs(left_eye_area, left_keypoints)

                left_eye_area = np.require(left_eye_area, np.uint8, 'C')
                self.display_image(left_eye_area, window='left')

        if self.pupilsCheckbox.isChecked():
            self.display_image(base_image)

    def get_keypoints(self, frame, frame_gray, threshold, previous_keypoint, previous_area):

        keypoints = self.process_eye(frame_gray, threshold, self.detector, prevArea=previous_area)
        if keypoints:
            previous_keypoint = keypoints
            previous_area = keypoints[0].size
        else:
            keypoints = previous_keypoint
        return keypoints, previous_keypoint, previous_area

    def display_image(self, img, window='main'):
        # Makes OpenCV images displayable on PyQT, displays them
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                qformat = QImage.Format_RGBA8888
            else:  # RGB
                qformat = QImage.Format_RGB888
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)  # BGR to RGB
        out_image = out_image.rgbSwapped()
        if window == 'main':  
            self.baseImage.setPixmap(QPixmap.fromImage(out_image))
            self.baseImage.setScaledContents(True)
        if window == 'left': 
            self.leftEyeBox.setPixmap(QPixmap.fromImage(out_image))
            self.leftEyeBox.setScaledContents(True)
        if window == 'right': 
            self.rightEyeBox.setPixmap(QPixmap.fromImage(out_image))
            self.rightEyeBox.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("Eye Tracking")
    window.show()
    sys.exit(app.exec_())

