import cv2
import numpy as np
from scipy.misc import imresize


draw_number_box_color = (255, 0, 255)


"""
Draw a box
"""
def draw_box(img, points, color):
    # cv2.line(img, (0,0), (100,100), color, 5)
    for i in range(len(points)):
        cv2.line(img, tuple(points[i]), tuple(points[(i+1)%(len(points))]), color, 10)



def general_number_extractor(src_img):
    # convert source iamge to gray scale and resize
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    gray = imresize(gray, [50, 80])

    # blur
    # gray = cv2.medianBlur(gray,13)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # threshold
    # ret, gray = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 15, 3)
    # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # bit wise inverse
    # gray = cv2.bitwise_not(gray)

    ######################
    kernel = np.ones([3, 3], np.uint8)
    gray = cv2.dilate(gray, kernel, iterations = 1)

    """
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = gray.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # print(mask)
    # print(gray)

    # Floodfill from point (0, 0)
    # cv2.floodFill(gray, mask, (50,0),255);

    # print(gray[49][0])

    gray = cv2.erode(gray, kernel, iterations = 1)
    """

    return gray



def rim_extractor(src_img, color):
    # select color that draw by us
    defined_color = np.array(list(color))
    gray = cv2.inRange(src_img, defined_color, defined_color)

    # resize
    gray = imresize(gray, [50, 80])

    # threshold
    ret, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # enlarge rim a little bit
    kernel = np.ones([5, 5], np.uint8)
    gray = cv2.dilate(gray, kernel, iterations = 1)


    return gray



def region_of_interest(img, box):
    x0, y0 = box[1]
    x1, y1 = box[3]
    # The following number should be in range:
    x0 = max(int(round(x0)), 0)
    x1 = min(int(round(x1)), img.shape[1])
    y0 = max(int(round(y0)), 0)
    y1 = min(int(round(y1)), img.shape[0])
    roi = np.copy(img[y0:y1, x0:x1])
    return roi



"""
This function will extract roi after the number boxes have been found
"""
def preprocess_for_number_recognition(src_img, rects, number_boxes):
    global draw_number_box_color
    number_boxes_regions_list = list()
    box_index = 0

    for box in number_boxes:

        # prepare for extracting process
        general_number_temp = general_number_extractor(region_of_interest(src_img, box))
        draw_box(src_img, box, draw_number_box_color) # draw the rim
        rim_temp = rim_extractor(region_of_interest(src_img, box), draw_number_box_color)

        box_center = rects[box_index][0]
        cv2.circle(src_img, (int(round(box_center[0])), int(round(box_center[1]))), 1, (0,0,255), 5)


        # extracting
        roi_temp = rim_temp + general_number_temp

        kernel = np.ones([3, 3], np.uint8)
        extracted_result = cv2.erode(roi_temp, kernel, iterations = 1)

        number_boxes_regions_list.append(extracted_result)

        # update loop variable
        box_index += 1

    return number_boxes_regions_list
