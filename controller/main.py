import os, sys
file_dir = os.path.dirname(os.path.abspath(__file__))
root = file_dir + '/..'
sys.path.insert(0, root)
from led_displays_searching_recognition.led import LedDisplaysRecognizer
from handwritten_digit_recognition.classifier import HandwrittenDigitClassifier
from number_searching.grid_recognition import number_search
from controller.communication import *
import cv2
import numpy as np
from scipy.misc import imresize
import time

class MainController:
    def __init__(self):
        self.communication = Communication()
        self.cam = cv2.VideoCapture(0)
        self.led_displays_recognizer = LedDisplaysRecognizer()

        # Use the right mode:
        self.handwritten_digit_classifier = HandwrittenDigitClassifier(mode = 'GPU')
        # self.handwritten_digit_classifier = HandwrittenDigitClassifier(mode = 'CPU')

        self.loop()

    def loop(self):
        """
        Control start/stop and communication with the driver:
        """
        while True:
            msg = self.communication.receive()
            if msg == MSG_START:
                self.communication.send([MSG_CONFIRM]) # confirmation respond
                while True:
                    x_axis, y_axis = self.process()
                    self.communication.send([MSG_AXIS]) # trigger driver state machine to receive axis
                    self.communication.send_axis(x_axis) # send x axis
                    self.communication.send_axis(y_axis) # send y axis
                    if self.communication.receive() == MSG_STOP:
                        # Suspend the program
                        break

    def process(self):
        """
        Process each frame, return the x-axis and y-axis errors of the target
        with respect to the center of the screen, scale the errors so that they
        are ranged [-128, 127], which will be send to the robot driver.

        If a target has been activated, then go to the next target.
        """
        # # cam = cv2.VideoCapture(0)
        # cam = cv2.VideoCapture('../test1.mpeg')
        # while True:
            # ret, frame = cam.read()
            # assert ret == True

            # h, w, _ = frame.shape
            # h2 = h / 2
            # w2 = w / 2
            # frame_ori = frame.copy()

            # result = led_displays_recognizer.process(frame)
            # if result != None:
                # digits, x0, x1, y0, y1 = result
                # cv2.putText(frame, digits, (x0, y0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))
            # else:
                # cv2.putText(frame, 'Not Found', (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))

            # src_img, number_boxes_regions_list, rects = number_search(frame)
            # for i in xrange(len(number_boxes_regions_list)):
                # roi = number_boxes_regions_list[i]
                # prob = handwritten_digit_classifier.predict(roi)
                # prediction = np.argmax(prob)
                # center_x, center_y = rects[i][0]
                # err_x = w2 - center_x
                # err_y = h2 - center_y
                # center_x = int(round(center_x))
                # center_y = int(round(center_y))
                # cv2.putText(frame, str(prediction), (center_x, center_y),
                        # cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

            # cv2.imshow('frame', src_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
        x_axis = 120
        y_axis = -90
        return x_axis, y_axis

    def close(self):
        self.cam.release()
        self.communication.close()
        self.cv2.destroyAllWindows()

if __name__ == '__main__':
    main_controller = MainController()
    main_controller.close()
