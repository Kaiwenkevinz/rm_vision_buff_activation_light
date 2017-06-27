import cv2, sys, os, rospy, math
import numpy as np
from scipy.misc import imresize

root = os.path.dirname(os.path.abspath(__file__))
root = root+'/..'#'/number_searching'
sys.path.insert(0, root)
# print(root)
print(os.path.dirname(root))
from number_searching.grid_recognition import read_image_from_file,preprocessing_for_number_searching,filter_redundancy_boxes
from number_searching.preprocess_for_number_recognition import draw_box, region_of_interest

file_dir = None
is_debug_mode = True

draw_prompt_lights_box_color = (255,255,255)

"""
Analysis and filter contours
"""
def analysis_and_filter_contours_for_prompt_lights_searching(contours):
    ratio = 2.0 / 1.0
    sudokuWidth = 50
    sudokuHeight = 25
    angleTolerance = 6
    ratioToleranceRate = 0.2
    dimensionsToleranceRate = 0.4

    contours_filtered = list()
    rects = list()
    boxes = list()
    for contour in contours:
        tempRect = cv2.minAreaRect(contour)
        # if is_debug_mode:
        #     print("[Debug] tempRect:", tempRect)

        width = tempRect[1][0]
        height = tempRect[1][1]

        if not (width > height):
            # tempRect = cv2.boxPoints((tempRect[0],(tempRect[1][0],tempRect[1][1]),tempRect[2] + 90.0))
            tempRect = (tempRect[0],(tempRect[1][1],tempRect[1][0]),tempRect[2] + 90.0)
            width = tempRect[1][0]
            height = tempRect[1][1]

        if(height==0):
            height = -1

        ratio_cur = width / height

        if (ratio_cur > (1.0-ratioToleranceRate) * ratio and \
            ratio_cur < (1.0+ratioToleranceRate) * ratio and \
			width > (1.0-dimensionsToleranceRate) * sudokuWidth and \
            width < (1.0+dimensionsToleranceRate) * sudokuWidth and \
			height > (1.0-dimensionsToleranceRate) * sudokuHeight and \
            height < (1.0+dimensionsToleranceRate) * sudokuHeight and \
			((tempRect[2] > -angleTolerance and tempRect[2] < angleTolerance) or \
              tempRect[2] < (-180+angleTolerance) or \
              tempRect[2] > (180-angleTolerance))
        ):
              contours_filtered.append(contour)
              rects.append(tempRect)
              if (is_debug_mode):
                  tempRect_points = cv2.boxPoints(tempRect)
                  boxes.append(tempRect_points)

    return contours_filtered, rects, boxes



"""
Pre-processing image
"""
def preprocessing_for_prompt_lights_searching(src_img):
    # convert source iamge to gray scale and resize
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # gray = imresize(gray, [50, 80])

    # blur
    # gray = cv2.medianBlur(gray,13)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # threshold
    # ret, gray = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 15, 3)
    # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # enhance outline
    kernel = np.ones([3, 3], np.uint8)
    gray = cv2.dilate(gray, kernel, iterations = 1)


    return gray



"""
This function first classify points into groups by x distance.
Then, pick up the larget group.
"""
def prompt_light_filter_outlier_boxes_by_x_dist(contours, rects, number_boxes):
    dist_list = [[rects[i]] for i in range(len(rects))]
    boxes_list = [[number_boxes[i]] for i in range(len(rects))]
    contours_list = [[contours[i]] for i in range(len(rects))]
    x_bin_size = 10

    # find near centre points for each centre point (by horizontal distance)
    for rect_i in range(len(rects)):
        for rect_j in range(rect_i+1,len(rects)):
            rect_i_center_x = rects[rect_i][0][0]
            rect_i_center_y = rects[rect_i][0][1]
            rect_j_center_x = rects[rect_j][0][0]
            rect_j_center_y = rects[rect_j][0][1]

            dist_x = abs(rect_i_center_x - rect_j_center_x)
            dist_y = abs(rect_i_center_y - rect_j_center_y)

            dist_ij = dist_x**2 + dist_y**2

            if dist_x < x_bin_size:
                dist_list[rect_i].append(rects[rect_j])
                dist_list[rect_j].append(rects[rect_i])
                boxes_list[rect_i].append(number_boxes[rect_j])
                boxes_list[rect_j].append(number_boxes[rect_i])
                contours_list[rect_i].append(contours[rect_j])
                contours_list[rect_j].append(contours[rect_i])

    # get the size of each bin
    dist_len_list = [0.0] * len(rects)
    for i in range(len(dist_list)):
        dist_len_list[i] = len(dist_list[i])

    # largest bin (group) size
    max_bin_size = max(dist_len_list)

    good_list_index = dist_len_list.index(max(dist_len_list))

    bad_box_indexs = list()
    good_contours = contours_list.pop(good_list_index)
    good_rects = dist_list.pop(good_list_index)
    good_boxes = boxes_list.pop(good_list_index)


    return good_contours, good_rects, good_boxes, bad_box_indexs



"""
This function get rid of outlier prompt light boxes
"""
def filter_outlier_boxes(contours, rects, number_boxes):
    dist_list = [0.0] * len(rects)

    for rect_i in range(len(rects)):
        for rect_j in range(rect_i+1,len(rects)):
            rect_i_center_x = rects[rect_i][0][0]
            rect_i_center_y = rects[rect_i][0][1]
            rect_j_center_x = rects[rect_j][0][0]
            rect_j_center_y = rects[rect_j][0][1]

            dist_x = abs(rect_i_center_x - rect_j_center_x)
            dist_y = abs(rect_i_center_y - rect_j_center_y)

            dist_ij = dist_x**2 + dist_y**2

            dist_list[rect_i] += dist_ij
            dist_list[rect_j] += dist_ij

    bad_box_indexs = list()
    good_contours = list()
    good_rects = list()
    good_boxes = list()
    for i in range(min(5, len(rects))):
        current_min_index = dist_list.index(min(dist_list))

        bad_box_indexs.append(dist_list.pop(current_min_index))
        good_contours.append(contours.pop(current_min_index))
        good_rects.append(rects.pop(current_min_index))
        good_boxes.append(number_boxes.pop(current_min_index))

    return good_contours, good_rects, good_boxes, bad_box_indexs



"""
This function will extract roi after the prompt lights have been found
"""
def preprocess_for_prompt_light_identify(src_img, rects, number_boxes):
    global draw_prompt_lights_box_color
    number_boxes_regions_list = list()
    box_index = 0
    # src_img = cv2.GaussianBlur(src_img,(51,51),0)

    for box in number_boxes:

        # extract ROI to pick the most comment color in the box
        blur = cv2.GaussianBlur(region_of_interest(src_img, box),(15,15),0)
        blur = imresize(blur, [25, 50]) # resize

        # simply get rid of rim
        draw_prompt_lights_box_color = (int(blur[(12,25)][0]),int(blur[(12,25)][1]),int(blur[(12,25)][2]))
        draw_box(src_img, box, draw_prompt_lights_box_color) # draw the rim with a most comment color in the box

        # extract ROI for promt lights identify
        blur = cv2.GaussianBlur(region_of_interest(src_img, box),(15,15),0)
        extracted_result = imresize(blur, [25, 50]) # resize

        # extracted result ready for return
        number_boxes_regions_list.append(extracted_result)

        # Debug
        box_center = rects[box_index][0]
        cv2.circle(src_img, (int(round(box_center[0])), int(round(box_center[1]))), 1, (0,0,255), 5)

        # update loop variable
        box_index += 1

    return number_boxes_regions_list



"""
Major process of prompt lights searching
"""
def prompt_lights_searching(src_img):
    processed_img = preprocessing_for_prompt_lights_searching(src_img)
    # processed_img = preprocessing_for_number_searching(src_img)
    # src_img = np.copy(processed_img)
    im2, contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(src_img, contours, -1, (255,0,0), 3)
    # cv2.fillPoly(src_img, contours, (0, 255, 0))
    # print(type(contours))

    #Analysis to get boxes
    contours, rects, number_boxes = analysis_and_filter_contours_for_prompt_lights_searching(contours)
    # cv2.drawContours(src_img, contours, -1, (255,0,255), 3)

    #Avoid redundancy boxes
    contours, rects, number_boxes, _ = filter_redundancy_boxes(contours, rects, number_boxes)

    #Find a largest bin in x direction
    contours, rects, number_boxes, _ = prompt_light_filter_outlier_boxes_by_x_dist(contours, rects, number_boxes)

    #Avoid outliers
    _, rects, number_boxes, _ = filter_outlier_boxes(contours, rects, number_boxes)

    #Extract info for prompt lights identify
    number_boxes_regions_list = preprocess_for_prompt_light_identify(src_img, rects, number_boxes)

    if len(rects) == 5:
        for i in range(len(rects)):
            draw_box(src_img, number_boxes[i], (0,255,0)) # draw the rim
            cv2.putText(src_img, str(number_boxes_regions_list[i][(12,25)]), (int(rects[i][0][0]),int(rects[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # print(src_img[rects[i][0]])
    # print("+++++++++++++")

    for i in range(len(contours)):
        cv2.drawContours(src_img, contours, i, (255,0,0),3)
        # cv2.fillPoly(src_img, list(contours[i]), (0,255,0))

        # cv2.imshow("kankan", src_img)
        # key = cv2.waitKey(0) & 0xff
        # if key == ord('q'):
        #     break

    return src_img, number_boxes_regions_list



"""
Main function (for testing)
"""
if __name__ == "__main__":
    """ ================ Testing with image files (START) ================ """
    """
    # import .grid_recognition
    # from grid_recognition import read_image_from_file


    #load src image
    src_img = read_image_from_file()
    # src_img, number_boxes_regions_list, _ = number_search(src_img)
    src_img = prompt_lights_searching(src_img)

    cv2.imshow('src_img', src_img)
    key = cv2.waitKey(0)

    """
    """ ================= Testing with image files (END) ================= """

    """ ================ Testing with video files (START) ================ """
    # """
    # cam = cv2.VideoCapture('./../Buff2017.mp4')
    cam = cv2.VideoCapture('./../../buff_test_video_01.mpeg')

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = None#cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    frame_num = 1
    segment_num = 1
    frame_rate = 24
    recording = False

    while True:
        ret, frame = cam.read()
        assert ret == True

        # src_img, number_boxes_regions_list, _ = number_search(frame)
        src_img, number_boxes_regions_list = prompt_lights_searching(frame)

        cv2.imshow('src_img', src_img)
        for i in range(len(number_boxes_regions_list)):
            cv2.imshow(str(i),number_boxes_regions_list[i])

        key = cv2.waitKey(1000/frame_rate) & 0xff
        # key = cv2.waitKey(0) & 0xff
        if key == ord('q'):
            break

    # """
    """ ================= Testing with image files (END) ================= """

    cv2.destroyAllWindows()
