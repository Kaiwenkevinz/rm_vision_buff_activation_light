import cv2, sys, rospy, math
import numpy as np
from scipy.misc import imresize

file_dir = None
is_debug_mode = True


"""
Parse the file direction from argv and load the image
"""
def read_image_from_file():

    # parse the file direction
    file_dir = None
    if(len(sys.argv)>1):
        file_dir = sys.argv[1]
    else:
        print("Error: No file name provided")
        exit()

    # load the image and return
    img = cv2.imread(file_dir)
    return img



"""
Pre-processing image
"""
def preprocessing_for_number_searching(src_img):
    # convert source iamge to gray scale
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray,(5,5),0)

    # threshold
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 15, 3)
    # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # blur
    gray = cv2.medianBlur(gray,3)
    # gray = cv2.GaussianBlur(gray,(5,5),0)

    return gray



"""
Analysis and filter contours
"""
def analysis_and_filter_contours_for_number_searching(contours):
    ratio = 28.0 / 16.0
    sudokuWidth = 66#127
    sudokuHeight = 38#71
    angleTolerance = 6
    ratioToleranceRate = 0.2
    dimensionsToleranceRate = 0.4

    num = 0

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
This function get rid of redundancy number boxes
"""
def filter_redundancy_boxes(contours, rects, number_boxes):
    bad_box_indexs = list()
    dist_toleration = 10

    for rect_i in range(len(rects)):
        if rect_i not in bad_box_indexs:
            for rect_j in range(rect_i+1, len(rects)):
                rect_i_center_x = rects[rect_i][0][0]
                rect_i_center_y = rects[rect_i][0][1]
                rect_j_center_x = rects[rect_j][0][0]
                rect_j_center_y = rects[rect_j][0][1]

                dist_x = abs(rect_i_center_x - rect_j_center_x)
                dist_y = abs(rect_i_center_y - rect_j_center_y)

                dist_ij = dist_x**2 + dist_y**2

                if dist_ij < dist_toleration**2:
                    rect_i_area = rects[rect_i][1][0] * rects[rect_i][1][1]
                    rect_j_area = rects[rect_j][1][0] * rects[rect_j][1][1]

                    if rect_i_area < rect_j_area:
                        bad_box_indexs.append(rect_j)
                    else:
                        bad_box_indexs.append(rect_i)

    good_contours = list()
    good_rects = list()
    good_boxes = list()
    for i in range(len(number_boxes)):
        if i not in bad_box_indexs:
            good_contours.append(contours[i])
            good_rects.append(rects[i])
            good_boxes.append(number_boxes[i])

    return good_contours, good_rects, good_boxes, bad_box_indexs



"""
This function get rid of outlier number boxes
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

    # print min(dist_list)

    bad_box_indexs = list()
    good_contours = list()
    good_rects = list()
    good_boxes = list()
    for i in range(min(9, len(rects))):
        current_min_index = dist_list.index(min(dist_list))

        bad_box_indexs.append(dist_list.pop(current_min_index))
        good_contours.append(contours.pop(current_min_index))
        good_rects.append(rects.pop(current_min_index))
        good_boxes.append(number_boxes.pop(current_min_index))

    return good_contours, good_rects, good_boxes, bad_box_indexs



"""
Major structure here
"""
def number_search(src_img):

    #preprocessing image
    processed_img = preprocessing_for_number_searching(src_img)
    # processed_img = preprocessing_for_number_recognition(src_img)
    # cv2.imshow('processed_img', processed_img)
    # cv2.waitKey(0)

    #find contours
    im2, contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(src_img, contours, -1, (255,0,0), 3)
    # print(contours[0])

    #Analysis to get boxes
    contours, rects, number_boxes = analysis_and_filter_contours_for_number_searching(contours)
    # cv2.drawContours(src_img, contours, -1, (255,0,255), 3)

    #Avoid redundancy boxes
    contours, rects, number_boxes, _ = filter_redundancy_boxes(contours, rects, number_boxes)
    # print(bad_box_indexs)
    # print(number_boxes)

    #Avoid outliers
    contours, rects, number_boxes, _ = filter_outlier_boxes(contours, rects, number_boxes)

    #Extracting infomation for number recognition
    from preprocess_for_number_recognition import preprocess_for_number_recognition
    number_boxes_regions_list = preprocess_for_number_recognition(src_img, rects, number_boxes)

    if is_debug_mode:
        print("The size of filtered contour list is:", len(contours))
        # print(cv2.minAreaRect(contours[0]))

    return src_img, number_boxes_regions_list, rects


"""
Main function (for testing)
"""
if __name__ == "__main__":
    """ ================ Testing with image files (START) ================ """
    """
    from matplotlib import pyplot as plt

    #load src image
    src_img = read_image_from_file()
    src_img, number_boxes_regions_list, _ = number_search(src_img)

    cv2.imshow('src_img', src_img)


    for i in range(len(number_boxes_regions_list)):
        cv2.imshow("single_test"+str(i), number_boxes_regions_list[i])
        cv2.waitKey(0)

        plt.subplot(3,3,i+1),plt.imshow(number_boxes_regions_list[i],'gray')
        plt.title(str(i))
        # plt.xticks([]),plt.yticks([])

        # cv2.imshow(str(i), number_boxes_regions_list[i])
        # np.stack()

    plt.show()
    key = cv2.waitKey(0)

    """
    """ ================= Testing with image files (END) ================= """

    """ ================ Testing with video files (START) ================ """
    # """
    # cam = cv2.VideoCapture('./../Buff2017.mp4')
    cam = cv2.VideoCapture('../../buff_test_video_00.mpeg')
    # cam = cv2.VideoCapture(1)

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

        src_img, number_boxes_regions_list, _ = number_search(frame)

        cv2.imshow('src_img', src_img)
        for i in range(len(number_boxes_regions_list)):
            cv2.imshow(str(i),number_boxes_regions_list[i])
            # np.stack()

        key = cv2.waitKey(1000/frame_rate) & 0xff
        if key == ord('q'):
            break
    # """
    """ ================= Testing with image files (END) ================= """

    cv2.destroyAllWindows()
