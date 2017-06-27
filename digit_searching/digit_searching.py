import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #plt.imshow(mask)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def rgb_select(image, thresh=(0, 255)):
    red_channel = image[:,:,0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    
    binary_output = np.zeros_like(red_channel)
    binary_output[(red_channel - 0.5*green_channel - 0.5*blue_channel > thresh[0]) & ( red_channel - 0.5*green_channel - 0.5*blue_channel<= thresh[1])] = 255
    return binary_output

def gray_threshold(image, thresh=(0, 255)):
    binary_output = np.zeros_like(image)
    binary_output[(image > thresh[0]) & (image <= thresh[1])] = 255
    return binary_output

def ssr(five_heights):
    error = five_heights - five_heights.mean()
    return np.sum(error ** 2)

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes

def preprocess(image):
    image = cv2.resize(image, (1280, 720))
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # apply a mask, only search the central area
    masked_image = region_of_interest(image, VERTICES)
    
    # combine two binary images
    rgb_binary = rgb_select(masked_image, thresh=(40, 255))
    gray_binary = gray_threshold(cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY), thresh=(100, 255))
    
    combined = np.zeros_like(rgb_binary)
    combined[(rgb_binary == 255) & (gray_binary == 255)] = 255
    
    return image, rgb_binary, combined

def five_cloest(keypoints):
    # sort heights
    heights = []
    for kpt in keypoints:
        heights.append(kpt.pt[1])
    heights = np.array(heights)
    sorted_heights = heights.copy()
    sorted_heights.sort()
    
    # find cloest
    min_sum = 99999
    min_idx = None
    for i in range(0, len(sorted_heights)-4):
        now_sum = ssr(sorted_heights[i:i+5])
        if now_sum < min_sum:
            min_sum = now_sum
            min_idx = i
    
    # store cloest
    keypoints_copy = keypoints.copy()
    keypoints = []
    if min_idx == None:
        min_idx = 0
    for num in sorted_heights[min_idx:min_idx+5]:
        index = list(heights).index(num)
        keypoints.append(keypoints_copy[index])
    
    return keypoints

def region_of_digits(keypoints):
    # new heights and widths
    heights = []
    widths = []
    for keypoint in keypoints:
        #print(keypoint.pt, keypoint.size)
        heights.append(keypoint.pt[1])
        widths.append(keypoint.pt[0])
    sorted_widths = widths.copy()
    sorted_widths.sort()
    # black out surroundings
    average_height = np.array(heights).mean()
    average_width = np.array(widths).mean()
    my_sum = 0
    for i in range(len(sorted_widths)-1):
        my_sum += sorted_widths[i+1] - sorted_widths[i] 
    average_gap = my_sum / 4
    height = average_gap * 1.5
    
    y1 = int(average_height - height / 2)
    y2 = int(average_height + height / 2)
    x1 = int(average_width - average_gap*2.6)
    x2 = int(average_width + average_gap*2.6)
    
    return x1, x2, y1, y2

def five_biggest(cnts):
    digitCnts = []
    areas = []
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        areas.append(w*h)

    sorted_areas = areas.copy()
    sorted_areas.sort()
    # five biggest
    for area in sorted_areas[-5:]:
        digitCnts.append(cnts[areas.index(area)])
    # sort it left to right
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]
    
    return digitCnts

def identify_digits(digitCnts, image_binary, output, debug, h_w_ratio=2.5, area_ratio=0.4):
    digits = []
    # loop over each of the digits
    for i, c in enumerate(digitCnts):
        #print(i)
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        if h / w > h_w_ratio:  # special case: 1
            digit = 1
            digits.append(digit)
            if debug:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 10)
        else:
            roi = image_binary[y:y + h, x:x + w]

            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)

            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)), # top
                ((0, 0), (dW, h // 2)), # top-left
                ((w - dW, 0), (w, h // 2)), # top-right
                ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
                ((0, h // 2), (dW, h)), # bottom-left
                ((w - dW, h // 2), (w, h)), # bottom-right
                ((0, h - dH), (w, h)) # bottom
                ]
            on = [0] * len(segments)

            # loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)

                # if the total number of non-zero pixels is greater than
                # 40% of the area, mark the segment as "on"
                if total / float(area) > area_ratio:
                    on[i]= 1

            # lookup the digit and draw it on the image
            if debug:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 10)
            try:
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit) 
            except:
                digit = 11
                digits.append(digit)
    if debug:
        for num, digit in enumerate(digits):
            cv2.putText(output, str(digit), (200 + num * 300, y + 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
    else:
#         print(my_digits.result_digits, digits)
        my_digits.update(digits)
#         print(my_digits.result_digits)
        for num, digit in enumerate(my_digits.result_digits):
            if digit == 11:
                cv2.putText(output, "=", (520 + num * 50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
            else:
                cv2.putText(output, str(digit), (520 + num * 50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
#     print(my_digits.stored_digits)
#     print(my_digits.result_digits)
    
    return output

class FiveDigits():
    def __init__(self):
        self.frame_change = True
                            # 0   1   2   3   4
        self.result_digits = [11, 11, 11, 11, 11]
                            #  new             old
        self.stored_digits = [[],
                              [],
                              [],
                              [],
                              []]
        self.last_digits = [11, 11, 11, 11, 11]
    def update(self, digits):
        num_changed = 0
        for i in range(5):
            if (digits[i] != self.last_digits[i]) & (digits[i] != 11) & (self.last_digits[i] != 11):
                num_changed += 1
        if num_changed > 2:
            self.frame_change = True
#         if (np.array(digits) == np.array(self.last_digits)).sum() < 2:
#             self.frame_change = True
        self.last_digits = digits
        if self.frame_change:
#             print("changed!!!!!")
            # if frame has changed, just add them
            self.result_digits = digits
            self.stored_digits = [[digits[0]],
                                  [digits[1]],
                                  [digits[2]],
                                  [digits[3]],
                                  [digits[4]]]
            self.frame_change = False
        else:
#             print("before", self.result_digits)
#             print(digits)
            # if not, update stored digits
            for i, digit in enumerate(digits):
                if digit != 11:
                    # identified
                    if len(self.stored_digits[i]) == 5:
                        self.stored_digits[i].pop()
                    self.stored_digits[i] = [digit] + self.stored_digits[i]
                    # the dominating digits are result
#                     print(self.result_digits[i], max(set(self.stored_digits[i]), key=self.stored_digits.count))
                self.result_digits[i] = max(set(self.stored_digits[i]), key=self.stored_digits[i].count)
#             print(self.stored_digits)
#             print("after", self.result_digits)

def process_image(image, debug=False):
    # preprocess image
    image, rgb_binary, combined = preprocess(image)
    
    # find blobs by color
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(rgb_binary)
    if len(keypoints) < 5:
        cv2.putText(output, "ERROR 1", (520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        return output
    
    # only five cloest keypoints remain
    keypoints = five_cloest(keypoints)
    if len(keypoints) < 5:
        cv2.putText(output, "ERROR 1.2", (520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        return output
    
    # find the region to identify
    x1, x2, y1, y2 = region_of_digits(keypoints)
    #vertices = np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]], dtype=np.int32)
    # masked_image = region_of_interest(image, vertices)
    
    # create output image
    if debug:
        output = resize(image[y1:y2, x1:x2].copy(), height=500)
    else:
        output = image.copy()
    
    # noise cancelling
    try:
        image_binary = resize(combined[y1:y2, x1:x2], height=500)
    except:
        cv2.putText(output, "ERROR 2", (520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        return output
    image_binary = cv2.erode(image_binary, np.ones((5,5), np.uint8), iterations=5)
    image_binary = cv2.dilate(image_binary, np.ones((5,5), np.uint8), iterations=5)
    
    # rotate a little bit
    rows,cols = image_binary.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),5,1)
    image_binary = cv2.warpAffine(image_binary,M,(cols,rows))
    if debug:
        output = cv2.warpAffine(output,M,(cols,rows))
    
    # find contours
    _, cnts, _ = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if cnts == []:
        # fail to detect cnts
        cv2.putText(output, "ERROR 3", (520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        return output
    
    if len(cnts) < 5:
        cv2.putText(output, "ERROR 3.2", (520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        return output
    # only five biggest conts remain
    digitCnts = five_biggest(cnts)
    
    # identify digits in this region
    output = identify_digits(digitCnts, image_binary, output, debug=debug, h_w_ratio=2.5, area_ratio=0.4)

    return output

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 255;
 
# Filter by Area.
params.filterByArea = False
params.minArea = 100
params.maxArea = 2000
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.01
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.8
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
params.maxInertiaRatio = 0.3

params.filterByColor = False
params.blobColor = 255

DIGITS_LOOKUP = {
    #0  1  2  3  4  5  6
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (0, 1, 0, 0, 1, 0, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 0, 1, 1, 0, 1, 1): 3,
    (1, 0, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 0, 1, 0, 0, 0, 0): 7,
    (1, 0, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 0, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

VERTICES = np.array([[(300, 520), (300, 170), (980, 170), (980, 520)]], dtype=np.int32)

if __name__ ==  "__main__":
    #print("!!!!")
    my_digits = FiveDigits()
    video_output1 = 'video_output.mp4'
    video_input1 = VideoFileClip('test_video.mp4')
    processed_video = video_input1.fl_image(process_image)
    processed_video.write_videofile(video_output1, audio=False)
