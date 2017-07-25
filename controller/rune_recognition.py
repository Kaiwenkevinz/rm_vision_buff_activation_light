import os, sys
file_dir = os.path.dirname(os.path.abspath(__file__))
root = file_dir + '/..'
sys.path.insert(0, root)
from led_displays_searching_recognition.led import LedDisplaysRecognizer
from handwritten_digit_recognition.classifier import HandwrittenDigitClassifier
from number_searching.grid_recognition import number_search
from prompt_lights import prompt_lights_searching
import cv2
import numpy as np
from scipy.misc import imresize

class RuneRecognition:
    def __init__(self, show_image = False, crop = None):
        """
        Apply all recognition algorithms in this module.
        """
        # self.cam = cv2.VideoCapture(0)
        self.cam = cv2.VideoCapture('../test1.mpeg')
        self.led_displays_recognizer = LedDisplaysRecognizer()

        # Use the right mode:
        self.handwritten_digit_classifier = HandwrittenDigitClassifier(mode = 'GPU')
        # self.handwritten_digit_classifier = HandwrittenDigitClassifier(mode = 'CPU')

        self.show_image = show_image
        self.crop = crop

    def process(self):
        """
        Process each frame, return the x-axis and y-axis errors of the target
        with respect to the center of the screen, scale the errors so that they
        are ranged [-128, 127], which will be send to the robot driver.

        If a target has been activated, then go to the next target.
        """
        ret, frame = self.cam.read()
        assert ret == True

        # Do the processing

        if self.show_image:
            cv2.imshow('Rune Recognition', frame_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return err_x, err_y

    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rune_recognition = RuneRecognition(show_image = True, crop = [0.4, 0.4])
    while True:
        rune_recognition.process()
    rune_recognition.close()
