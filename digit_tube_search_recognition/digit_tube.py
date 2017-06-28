import cv2
import numpy as np
from scipy.misc import imresize

def threshold(img, thresh):
    img[img >= thresh] = 255
    img[img < thresh] = 0
    return img

def get_result(candidate):
    result = []
    for each in candidate:
        digit = each[0][0] + 1
        result.append(digit)
    return result

def find_repeat(lst):
    for i in xrange(len(lst)):
        for j in xrange(i+1, len(lst)):
            if lst[i] == lst[j]:
                return i, j
    return None

class DigitTubeRecognizer:
    def __init__(self):
        self.load_templates()

    def load_templates(self):
        self.templates = {}
        for i in range(1, 10):
            img = cv2.imread('templates/%d.png'%(i), 0)
            self.templates[i] = img

    def resize_templates(self, out_h):
        for i, img in self.templates.items():
            h, w = img.shape
            out_w = w * (out_h / float(h))
            out_w = int(round(out_w))
            self.templates[i] = imresize(img, [out_h, out_w])

    def process(self, img):
        # Select the digit tube pixels from the image
        red_mask = cv2.inRange(img, np.array([0, 0, 200]), np.array([255, 255, 255]))
        black_mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([30, 255, 255]))
        black_mask = cv2.blur(black_mask, (10, 10))
        raw_segment = red_mask & black_mask
        segment = raw_segment.copy()

        # Remove small regions that are not likely be our target
        mass = cv2.blur(segment, (70, 70))
        mass = threshold(mass, 2)
        segment = threshold(segment, 2)
        segment = segment & mass

        # Compute the center
        M = cv2.moments(segment)
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        
        # Compute the range
        y_loc, x_loc = np.where(segment != 0)
        x0 = x_loc.min()
        y0 = y_loc.min()
        x1 = x_loc.max()
        y1 = y_loc.max()
        
        # Compute the area for all digits
        width_all = x1 - x0
        height_all = y1 - y0

        # If the area we got is too large, then there are possibly light
        # interferences, so we discard this image. This is not the perfect
        # solution.
        h, w = raw_segment.shape
        if width_all > w*0.2 or height_all > h*0.2:
            return None
        else:
            # Prepare for template matching
            width_each = int(round(width_all / 5.0))
            self.resize_templates(height_all)
            offset = 10
            x0 = max(x0 - offset, 0)
            y0 = max(y0 - offset, 0)
            x1 = min(x1 + offset, w)
            y1 = min(y1 + offset, h)
            segment = raw_segment[y0:y1, x0:x1]
            segment = 255 - segment
            scores = []

            # Apply template matching
            for num, template in self.templates.items():
                res = cv2.matchTemplate(segment, template, cv2.TM_CCOEFF_NORMED)
                scores.append(res)
            scores = np.asarray(scores)

            # Select some candidates that has the normalized score > 0.5
            candidate = []
            for i in xrange(5):
                bound = i * width_each
                region = scores[:, :, bound:bound+width_each]
                pts = zip(*np.where(region > 0.5))
                pts = sorted(
                    pts,
                    key = lambda x: region[x],
                    reverse = True
                )
                for j in xrange(len(pts)):
                    verify = region[pts[j]]
                    ch, h, w = pts[j]
                    pts[j] = ch, h, w + bound
                    assert scores[pts[j]] == verify
                candidate.append(pts)

            # Check repeated digits, if found then we only take the one with
            # higher score value
            result = get_result(candidate)
            while True:
                repeat = find_repeat(result)
                if repeat == None:
                    break
                i, j = repeat
                pt1 = candidate[i][0]
                pt2 = candidate[j][0]
                if scores[pt1] >= scores[pt2]:
                    candidate[j].pop(0)
                else:
                    candidate[i].pop(0)
                result = get_result(candidate)

            result = [str(x) for x in result]
            result = ''.join(result)
            return result, x0, x1, y0, y1


if __name__ == '__main__':
    cam = cv2.VideoCapture('../buff_test_video_00.mpeg')
    digit_tube_recognizer = DigitTubeRecognizer()

    while True:
        ret, frame = cam.read()
        assert ret == True

        h, w, _ = frame.shape
        rh = int(round(h * 0.2))
        rw = int(round(w * 0.2))
        result = digit_tube_recognizer.process(frame[rh:h-rh, rw:w-rw, :])
        if result != None:
            digits, x0, x1, y0, y1 = result
            x0 += rw
            y0 += rh

            cv2.putText(frame, digits, (x0, y0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))
        else:
            cv2.putText(frame, 'Not Found', (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))

        cv2.imshow('frame', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
