import argparse
import os

import cv2 as cv
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours


class grade_test(object):
    sorted_answer_key = []

    def readTextFile(self, textPath):
        linear = 0
        input_file = open(textPath, 'r')
        grade_test.unsorted_answer_key = input_file.read().split('\n')[:-1]
        for i in grade_test.unsorted_answer_key:
            print(i)
            x = str(i.split(",")[1])
            print(x)
            if (x == 'a'):
                grade_test.unsorted_answer_key[linear] = 0
            elif (x == 'b'):
                grade_test.unsorted_answer_key[linear] = 1
            elif (x == 'c'):
                grade_test.unsorted_answer_key[linear] = 2
            elif (x == 'd'):
                grade_test.unsorted_answer_key[linear] = 3
            linear += 1
        print(grade_test.unsorted_answer_key)

    def grade(self, image_path):
        # default_image = os.path.join("C:\\", "Users", "salma", "Desktop", "SPL-1", "main", "../Images", "test20.png")
        default_image = image_path
        parser = argparse.ArgumentParser()  # take command line arguments
        parser.add_argument("-i", "--image", required=False, default=default_image,
                            help="Please specify file path to the OMR sheet")  # image contains file path, required variable ensures command line input and help shows USAGE
        args = vars(parser.parse_args())  # store the filepath from argument parsed in command line

        # Please enter the values of the correct answers here

        ############################################################
        img = cv.imread(args["image"])
        # img = cv.resize(img, (1920, 1080))
        cv.imshow("or", img)
        cv.waitKey(0)
        blank = np.zeros(img.shape[:2], dtype='uint8')
        output = np.zeros(img.shape[:2], dtype='uint8')
        # Convert to Grayscale
        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply grayscale to Blur and reduce high frequency noise
        blurred_img = cv.GaussianBlur(grayscale, (3, 3),
                                      cv.BORDER_DEFAULT)  # the second parameter is kernel size and it is always odd tuple
        # Apply blurred image to detect edges
        edged = cv.Canny(blurred_img, 25, 80)  # 2nd and 3rd parameter are threshold values

        # Detect Contours
        # cnts, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        ############################# Find Largest Rectangle #######################################
        def find_contour_areas(contours):
            areas = []
            for cnt in contours:
                cont_area = cv.contourArea(cnt)
                areas.append(cont_area)
            return areas

        thresh = cv.threshold(blurred_img, 0, 255,
                              cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        cnts = cv.findContours(thresh.copy(), cv.RETR_LIST,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour, then use the
            # bounding box to derive the aspect ratio
            (x, y, w, h) = cv.boundingRect(c)
            ar = w / float(h)
            # print(ar, w, h)
            if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
                questionCnts.append(c)

        # sort the question contours top-to-bottom, then initialize
        # the total number of correct answer
        questionCnts = contours.sort_contours(questionCnts,
                                              method="top-to-bottom")[0]

        correct = 0
        newCnts = []
        for i in range(19):
            temp = questionCnts[i * 12:(i + 1) * 12]
            temp = temp[::-1]
            newCnts += temp
        questionCnts = newCnts
        answers = [""] * 100
        row = 0
        column = 0
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
            # sort the contours for the current question from
            # left to right, then initialize the index of the
            # bubbled answer
            print(q)
            cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
            bubbled = None
            for i, j in enumerate(cnts):
                # construct a mask that reveals only the current
                # "bubble" for the question
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv.drawContours(mask, [j], -1, 255, -1)
                # apply the mask to the thresholded image, then
                # count the number of non-zero pixels in the
                # bubble area

                mask = cv.bitwise_and(thresh, thresh, mask=mask)
                # cv.imshow("Circles", mask)
                # cv.waitKey(0)
                total = cv.countNonZero(mask)
                # if the current total has a larger number of total
                # non-zero pixels, then we are examining the currently
                # bubbled-in answer
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, i)
            print(bubbled)
            if (q != 0 and q % 3 == 0):
                row += 1
                column = row
            answers[column] = bubbled[1]
            print("COLUMN NO", column)
            column = column + 19

        for q in range(len(grade_test.unsorted_answer_key)):
            color = (0, 0, 255)
            k = grade_test.unsorted_answer_key[q]
            # check to see if the bubbled answer is correct
            if k == answers[q]:
                color = (0, 255, 0)
                correct += 1
            # draw the outline of the correct answer on the test
            cv.drawContours(img, [cnts[k]], -1, color, 3)

        print("[INFO] score: {:.2f}".format(correct))
        cv.putText(img, "{:.2f}".format(correct), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # loop over the sorted contours

        cv.drawContours(img, questionCnts, -1, (0, 255, 0), 2)
        cv.imshow("Exam", img)
        cv.waitKey(0)
        # cv.imshow('Circles Detected', masked)
        # cv.waitKey(0)
        cv.destroyAllWindows()
