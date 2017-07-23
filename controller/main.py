import os, sys
file_dir = os.path.dirname(os.path.abspath(__file__))
root = file_dir + '/..'
sys.path.insert(0, root)
from led_displays_searching_recognition.led import LedDisplaysRecognizer
from handwritten_digit_recognition.classifier import HandwrittenDigitClassifier
from number_searching.grid_recognition import number_search
from controller.communication import Communication
import cv2
import numpy as np
from scipy.misc import imresize

class MainController:
    def __init__(self):
        self.communication = Communication()
        self.cam = cv2.VideoCapture(0)
        self.led_displays_recognizer = LedDisplaysRecognizer()
        self.handwritten_digit_classifier = HandwrittenDigitClassifier()
        self.loop()

    def loop(self):
        # Control start/stop of the main program
        while True:
            if self.communication.start_received():
                self.communication.start_confirm()
                while True:
                    self.run()
                    if self.communication.stop_received():
                        break

    def run(self):
        # The main code goes here:
        self.communication.send([128, 129])

if __name__ == '__main__':
    main_controller = MainController()

    # # cam = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture('../test1.mpeg')
    # led_displays_recognizer = LedDisplaysRecognizer()
    # handwritten_digit_classifier = HandwrittenDigitClassifier()
    # communication = Communication()

    # while True:
        # msg = communication.receive()
        # # Check if start command is received

        # ret, frame = cam.read()
        # # frame = imresize(frame, [540, 816])
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
    # cam.release()
    # cv2.destroyAllWindows()
    # communication.close()
