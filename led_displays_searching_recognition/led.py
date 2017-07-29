import cv2, os
import numpy as np
from scipy.misc import imresize

def threshold(img, thresh):
    img[img >= thresh] = 255
    img[img < thresh] = 0
    return img

def get_result(candidate):
    result = []
    for scores in candidate:
        if len(scores) == 0:
            return []
        digit = scores[0][0] + 1
        result.append(digit)
    return result

def find_repeat(lst):
    for i in xrange(len(lst)):
        for j in xrange(i+1, len(lst)):
            if lst[i] == lst[j]:
                return i, j
    return None

class LedDisplaysRecognizer:
    def __init__(self):
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_templates()
        self.init_blob_detector()
        self.is_debug = True

    def load_templates(self):
        self.templates = {}
        for i in range(1, 10):
            img = cv2.imread(self.file_dir+'/templates/%d.png'%(i), 0)
            self.templates[i] = img

    def init_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 255

        params.filterByArea = False
        params.minArea = 1500

        params.filterByCircularity = False
        params.minCircularity = 0.1

        params.filterByConvexity = False
        params.minConvexity = 0.87

        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        params.filterByColor = False
        self.detector = cv2.SimpleBlobDetector_create(params)

    def resize_templates(self, out_h):
        templates = {}
        for i, img in self.templates.items():
            h, w = img.shape
            out_w = w * (out_h / float(h))
            out_w = int(round(out_w))
            templates[i] = imresize(img, [out_h, out_w])
        return templates

    def filter_blobs(self, img):
        """
        Remove all non interesting regions, and keep only one cluster which is the
        most promising region of interest. The idea is:

        Using opencv blob detection to cluster the image, after that, we can
        identify which cluster is more likely to be the region of interest.
        Specifically, here we think the LED digit cluster is the largest cluster
        whose height is way smaller than the width.

        The idea is inspired by @AlexQian97
        """
        filtered_img = img
        debug_img = img.copy()
        img_blured = cv2.blur(img, (60, 60))
        img_blured = threshold(img_blured, 2)
        keypoints = self.detector.detect(img_blured)
        if len(keypoints) != 0:
            keypoints = sorted(keypoints, key = lambda x: x.size, reverse = True)

            # Select a largest blob whose height is way smaller than the height:
            while len(keypoints) > 0:

                # Calculate the height and width:
                center_x, center_y = keypoints[0].pt
                r = keypoints[0].size
                center_x = int(round(center_x))
                center_y = int(round(center_y))
                r2 = int(round(r / 2.0))
                x0 = max(center_x - r2, 0)
                x1 = min(center_x + r2, img.shape[1])
                y0 = max(center_y - r2, 0)
                y1 = min(center_y + r2, img.shape[0])
                mask = np.zeros_like(img)
                mask[y0:y1, x0:x1] = 255
                segment = img & mask # segment the image
                y_loc, x_loc = np.where(segment != 0)
                if len(y_loc) == 0 or len(x_loc) == 0:
                    break
                segment_width = x_loc.max() - x_loc.min()
                segment_height = y_loc.max() - y_loc.min()

                if segment_width / segment_height > 2.0:
                    filtered_img = segment
                    if self.is_debug:
                        selected_pt = [keypoints[0]]
                        debug_img = cv2.drawKeypoints(img, selected_pt, np.array([]), (0, 0, 255),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    break
                keypoints.pop(0)

        if self.is_debug:
            cv2.imshow('Selected Cluster', debug_img)

        return filtered_img

    def process(self, img):
        # Select the digit tube pixels from the image
        red_mask = cv2.inRange(img, np.array([0, 0, 200]), np.array([255, 255, 255]))
        black_mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([30, 255, 255]))
        black_mask = cv2.blur(black_mask, (10, 10))
        segment = red_mask & black_mask
        segment_bkp = segment.copy()

        # Plan B:
        red_mask_extra = cv2.inRange(img, np.array([0, 0, 150]), np.array([90, 90, 255]))
        red_mask_extra = cv2.blur(red_mask_extra, (20, 20))
        red_mask_extra = threshold(red_mask_extra, 1)

        # Remove small regions that are not likely be our target
        mass = cv2.blur(segment, (70, 70))
        mass = threshold(mass, 2)
        segment = threshold(segment, 2)
        segment = segment & mass
        segment = segment & red_mask_extra
        segment = self.filter_blobs(segment)

        # Compute the center
        M = cv2.moments(segment)
        if M['m00'] == 0.0:
            return None
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])


        # Compute the range
        y_loc, x_loc = np.where(segment != 0)
        if len(x_loc) == 0 or len(y_loc) == 0:
            return None
        x0 = x_loc.min()
        y0 = y_loc.min()
        x1 = x_loc.max()
        y1 = y_loc.max()

        # Compute the area for all digits
        width_all = x1 - x0
        height_all = y1 - y0

        # If the area is too small we report not found.
        if width_all <= 0 or height_all <= 0:
            return None

        # If the area we got is too large, then there are possibly light
        # interferences, so we report not found. This is not a perfect
        # solution.
        # print height_all, width_all
        h, w = segment_bkp.shape
        # print w * 0.4, h * 0.4
        if width_all > w*0.4 or height_all > h*0.4:
            return None

        # Prepare for template matching
        width_each = int(round(width_all / 5.0))
        templates = self.resize_templates(height_all)
        offset = 10
        x0 = max(x0 - offset, 0)
        y0 = max(y0 - offset, 0)
        x1 = min(x1 + offset, w)
        y1 = min(y1 + offset, h)
        # segment = segment_bkp[y0:y1, x0:x1]
        segment = segment[y0:y1, x0:x1]
        segment = 255 - segment
        scores = []

        # Apply template matching
        for num, template in templates.items():
            if segment.shape[0] < template.shape[0] or segment.shape[1] < template.shape[1]:
                return None
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

        if len(result) == 0:
            return None

        result = [str(x) for x in result]
        result = ''.join(result)
        return result, x0, x1, y0, y1
