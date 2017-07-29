import os, sys
file_dir = os.path.dirname(os.path.abspath(__file__))
root = file_dir + '/..'
sys.path.insert(0, root)
from led_displays_searching_recognition.led import LedDisplaysRecognizer
from handwritten_digit_recognition.classifier import HandwrittenDigitClassifier
from number_searching.grid_recognition import number_search
from prompt_lights.prompt_lights_searching import prompt_lights_searching
import cv2
import numpy as np
from scipy.misc import imresize

class RuneRecognition:
    def __init__(self, show_image = False, crop = None):
        """
        Apply all recognition algorithms in this module.
        """
        # self.cam = cv2.VideoCapture(1)
        self.cam = cv2.VideoCapture(file_dir + '/../../../real_test_00.avi')
        # self.cam = cv2.VideoCapture(file_dir + '/../../preparation/real_test_00.avi')
        # self.cam = cv2.VideoCapture(file_dir + '/../../buff_test_video_01.mpeg')
        self.led_displays_recognizer = LedDisplaysRecognizer()

        # Use the right mode:
        # self.handwritten_digit_classifier = HandwrittenDigitClassifier(mode = 'GPU')
        self.handwritten_digit_classifier = HandwrittenDigitClassifier(mode = 'CPU')

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
            rh = int(round(h * ry_ratio)) # number of Y-axis pixels that will be cropped from original image
            rw = int(round(w * rx_ratio)) # number of X-axis pixels that will be cropped from original image
            roi_y0 = rh
            roi_y1 = h - rh
            roi_x0 = rw
            roi_x1 = w - rw
            if self.show_image:
                cv2.rectangle(frame_show, (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 255, 0), 2)
            roi = frame[roi_y0:roi_y1, roi_x0:roi_x1, :]
        else:
            rh = 0
            rw = 0
            roi = frame.copy()

        # Recognize LED digits:
        led_info = self.led_displays_recognizer.process(roi)
        if led_info != None:
            led_digits, x0, x1, y0, y1 = led_info
            x0 += rw # Convert cropped image axis to original axis
            x1 += rw
            y0 += rh
            y1 += rh
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
            prob = self.handwritten_digit_classifier.predict(digit_roi)
            prediction = np.argmax(prob)
            center_x, center_y = rects[i][0]
            center_x += rw # Convert cropped image axis to original axis
            center_y += rh
            handwritten_digit_locations[str(prediction)] = center_x, center_y
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            if self.show_image:
                cv2.putText(frame_show, str(prediction), (center_x, center_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        # Find how many prompt lights are activated:
        _, hitting_num = prompt_lights_searching(roi)
        if hitting_num >= 5:
            if self.show_image:
                cv2.imshow('Rune Recognition', frame_show)
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
                if self.show_image:
                    cv2.circle(frame_show, (int(round(target_x)), int(round(target_y))), 8, (0, 255, 255), -1)
                    cv2.putText(frame_show, 'err_x='+str(scaled_err_x)+' err_y'+str(scaled_err_y),
                            (300, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255))

        if self.show_image:
            cv2.imshow('Rune Recognition', frame_show)

        return target_err

    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rune_recognition = RuneRecognition(show_image = True, crop = [0.4, 0.4])
    # rune_recognition = RuneRecognition(show_image = True)
    while True:
        rune_recognition.process()
        if cv2.waitKey(1000/24) & 0xFF == ord('q'):
            break
    rune_recognition.close()
