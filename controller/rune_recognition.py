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

        frame_backup = frame.copy()
        frame_show = frame.copy()

        # Crop a region of interest (if applicable):
        if self.crop != None:
            rx_ratio = self.crop[0] * 0.5
            ry_ratio = self.crop[1] * 0.5
            h, w, _ = frame.shape
            rh = int(round(h * ry_ratio))
            rw = int(round(w * rx_ratio))
            roi_y0 = rh
            roi_y1 = h - rh
            roi_x0 = rw
            roi_x1 = w - rw
            if self.show_image:
                cv2.rectangle(frame_show, (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 255, 0), 2)
            roi = frame[roi_y0:roi_y1, roi_x0:roi_x1, :]
        else:
            roi = frame.copy()

        # Recognize LED digits:
        led_info = led_displays_recognizer.process(roi)
        if led_info != None:
            led_digits, x0, x1, y0, y1 = led_info
            if self.show_image:
                cv2.putText(frame_show, led_digits, (x0, y0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))
        else:
            if self.show_image:
                cv2.putText(frame_show, 'Not Found', (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))

        # Find handwritten digits and do the recognition:
        src_img, number_boxes_regions_list, rects = number_search(roi)
        handwritten_digit_locations = {}
        for i in xrange(len(number_boxes_regions_list)):
            digit_roi = number_boxes_regions_list[i]
            prob = handwritten_digit_classifier.predict(digit_roi)
            prediction = np.argmax(prob)
            center_x, center_y = rects[i][0]
            handwritten_digit_locations[prediction] = center_x, center_y
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            cv2.putText(frame_show, str(prediction), (center_x, center_y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        # Find how many prompt lights are activated:
        _, hitting_num = prompt_lights_searching(roi)
        if hitting_num >= 5:
            return None

        # Find the target based on informations collected above:
        target_err = None
        if led_info != None:
            h, w = frame.shape[:2]
            h2 = h * 0.5
            w2 = w * 0.5
            led_digits = led_info[0]
            target_digit = led_digits[hitting_num]
            if target_digit in handwritten_digit_locations:
                target_x, target_y = handwritten_digit_locations[target_digit]
                err_x = w2 - target_x
                err_y = h2 - target_y
                scaled_err_x = int(round(err_x / w2 * 127))
                scaled_err_y = int(round(err_y / h2 * 127))
                target_err = scaled_err_x, scaled_err_y
                if cv2.show_image:
                    cv2.circle(frame_show, (target_x, target_y), 5, (0, 255, 255))
                    cv2.putText(frame_show, 'err_x='+str(scaled_err_x)+' err_y'+str(scaled_err_y),
                            (100, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255))

        if self.show_image:
            cv2.imshow('Rune Recognition', frame_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return target_err

    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rune_recognition = RuneRecognition(show_image = True, crop = [0.4, 0.4])
    while True:
        rune_recognition.process()
    rune_recognition.close()
