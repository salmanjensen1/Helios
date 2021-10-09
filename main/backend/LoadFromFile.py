import argparse
import os

import cv2 as cv
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours


class grade_test(object):

    def grade(image_path):

        # default_image = os.path.join("C:\\", "Users", "salma", "Desktop", "SPL-1", "main", "../Images", "test20.png")
        default_image = image_path
        parser = argparse.ArgumentParser()  # take command line arguments
        parser.add_argument("-i", "--image", required=False, default=default_image,
                            help="Please specify file path to the OMR sheet")  # image contains file path, required variable ensures command line input and help shows USAGE
        args = vars(parser.parse_args())  # store the filepath from argument parsed in command line

        # Please enter the values of the correct answers here

        ############################################################
        img = cv.imread(args["image"])
        # img = cv.resize(img, (950, 1000))
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

        cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        docCnt = None
        # ensure that at least one contour was found
        if len(cnts) > 0:
            # sort the contours according to their size in
            # descending order
            cnts = sorted(cnts, key=cv.contourArea, reverse=True)
            # loop over the sorted contours
            for c in cnts:
                # approximate the contour
                peri = cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, 0.02 * peri, True)
                # if our approximated contour has four points,
                # then we can assume we have found the paper
                if len(approx) == 4:
                    docCnt = approx
                    break

        # perspective transform
        if docCnt is not None:
            paper = four_point_transform(img, docCnt.reshape(4, 2))
            img = paper
            warped = four_point_transform(blurred_img, docCnt.reshape(4, 2))
            thresh = cv.threshold(warped, 0, 255,
                                  cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        else:
            thresh = cv.threshold(blurred_img, 0, 255,
                                  cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        # # masked = cv.resize(masked, (900, 820))
        # thresh
        # thresh = cv.threshold(blurred_img, 0, 255,
        #                       cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
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
            if w >= 10 and h >= 10 and 0.9 <= ar <= 1.1:
                questionCnts.append(c)

        # sort the question contours top-to-bottom, then initialize
        # the total number of correct answers
        # cv.drawContours(img, questionCnts, -1, (0, 255, 0), 2)
        # print(len(questionCnts))

        # sort the question contours top-to-bottom, then initialize
        # the total number of correct answers

        questionCnts = contours.sort_contours(questionCnts,
                                              method="top-to-bottom")[0]
        # questionCnts = sorted(questionCnts, key=lambda ctr: cv.boundingRect(ctr)[0] + cv.boundingRect(ctr)[1] * thresh.shape[1] )

        # area = find_contour_areas(questionCnts)
        # print(area)

        correct = 0
        # # each question has 5 possible answers, to loop over the
        # # question in batches of 5
        # newCnts = []
        # for i in range(16):
        #     temp = questionCnts[i * 12:(i + 1) * 12]
        #     temp = temp[::-1]
        #     newCnts += temp
        # questionCnts = newCnts
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
            # sort the contours for the current question from
            # left to right, then initialize the index of the
            # bubbled answer
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
                cv.imshow("Circles", mask)
                cv.waitKey(0)
                total = cv.countNonZero(mask)
                # if the current total has a larger number of total
                # non-zero pixels, then we are examining the currently
                # bubbled-in answer
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, i)
            print(bubbled)
        # loop over the sorted contours

        cv.drawContours(img, questionCnts, -1, (0, 255, 0), 2)
        cv.imshow("Exam", img)
        cv.waitKey(0)
        # cv.imshow('Circles Detected', masked)
        # cv.waitKey(0)
        cv.destroyAllWindows()
