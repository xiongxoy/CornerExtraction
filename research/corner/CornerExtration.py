#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import copy
import logging

import cv2  # OpenCV 2
import numpy as np

from global_variable import GlobalVariable
from research.corner.image_transformer import ImageTransformer
from research.corner.rectangle_drawer import RectangleDrawer
from research.util import Plotter


class CornerExtractor:
    def __init__(self, image):
        self.image = copy.deepcopy(image)

    def extract(self, n=4, convex=True):
        return  self.get_bounding_polygon_vertices(self.image)

    def get_bounding_polygon_vertices(self, image):
        lines = self.get_lines(image)
        Plotter.plot_lines(image, lines)
        corners = self.get_vertices(lines)
        return corners

    def get_idx_from_contours_convex(self, contour, n=4):
        '''
        @param contour: contour in points, assumed to be convex
        @param n: number of sides
        '''
        assert isinstance(contour, np.ndarray)
        # calculate direction vectors
        k = []
        for i in range(contour.shape[0]):
            ki = contour[(i + 1) % contour.shape[0]] - contour[i]
            ki = ki.astype(np.float32)
            ki_mod = np.linalg.norm(ki, 2)
            ki[0] = ki[0] / ki_mod
            ki[1] = ki[1] / ki_mod
            k.append(ki)
        k = np.squeeze(np.vstack(k))

        # plot for debug
        image = np.zeros((300, 300))
        Plotter.plot_points(image, (k * 100 + 150))

        # use k-means
        termination_criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        _, labels, _ = cv2.kmeans(k, n,
                                  termination_criteria, 10,
                                  cv2.KMEANS_RANDOM_CENTERS)
        labels = list(np.squeeze(labels))
        return labels

    def get_idx_from_contours_concave(self, contours):
        '''
        @todo: 还没有完全实现
        @param contours:
        '''
        cv2.drawContours(self.image, [contours], 0, (0, 0, 255), 2)
        k = []
        for i in range(contours.shape[0]):
            # cv2.circle(image, tuple(contours[i][0]), 2, (255,255,0))
            ki = contours[(i + 1) % contours.shape[0]] - contours[i]
            ki = ki[0]
            ki = ki.astype(float)
            ki_mod = (ki[0] ** 2 + ki[1] ** 2) ** 0.5
            ki[0] = ki[0] / ki_mod
            ki[1] = ki[1] / ki_mod
            ki.append(contours[0], contours[1])
            k.append(ki)

        termination_criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        k_array = np.float32(k)
        centers = cv2.kmeans(k_array, 4,
                             termination_criteria, 10,
                             cv2.KMEANS_RANDOM_CENTERS)
        idx = centers[1]
        return contours, idx

        # build an array of direction+position ->
        #     what are the possible better choice?
        # get the index of each point
            # cluster the array, spectral clustering is a good choice
            # find lines in the 4d space
        # smooth the result using a 1*5 median filter
        raise Exception('To be implemented')

    def get_contour_from_image(self, image):
        # detect all edges in the image
        # convert to gray image
        bw_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur with mean value filter
        bw_img = cv2.blur(bw_img, (3, 3))
        # detect edges, sensitive to the parameter of canny
        bw_img = cv2.Canny(bw_img, 50, 150)
        Plotter.plot_image(bw_img, 'bw_img')

        # find contour of the edges
        # get contour
        contours, _ = cv2.findContours(copy.deepcopy(bw_img),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        # get the contour with the most points
        contour = reduce(lambda x, y: x if len(x) > len(y) else y,
                         contours, [])
        Plotter.plot_image(bw_img, 'bw_img_after_contour')
        Plotter.plot_contours(bw_img, [contour])
        contour = np.vstack(contour).squeeze()
        # plot contour
        Plotter.plot_points(bw_img, contour)
        # interpolate
#         contour = self.interpolate_points(contour)
        # plot contour
        Plotter.plot_points(bw_img, contour)
        Plotter.plot_contours(bw_img, [contour])
        return contour

    def get_lines_from_contour(self, contour):
        # get convex contour of a contour
        contour = np.squeeze(cv2.convexHull(contour))
        # assign points to different lines
        idx = self.get_idx_from_contours_convex(contour)
        # prepare points for line fitting
        idx = self.adjust_indexes(idx)
        # fit points to line
        lines = self.fit_lines_from_points(contour, idx)
        return lines

    def get_lines(self, image):
        '''
        @summary: extract boundary lines of polygon from image
        @param image: input image with polygon

        @return: extracted lines
        '''
        contour = self.get_contour_from_image(image)
        lines = self.get_lines_from_contour(contour)
        Plotter.plot_lines(image, lines)
        return lines

    def interpolate_points(self, contour, n=1):
        assert isinstance(contour, np.ndarray)
        for _ in xrange(n):
            contour_ret = []
            k = len(contour)
            for i in xrange(k):
                contour_ret.append(contour[i])
                contour_ret.append((contour[(i + 1) % k] + contour[i]) / 2)
            contour = contour_ret

        contour_ret = np.vstack(contour_ret)
        return contour_ret

    def get_indexes_in_window(self, indexes, i, w):
        assert isinstance(indexes, list)
        n = len(indexes)
        indexes_ret = []
        if i - w >= 0 and i + w + 1 <= n:
            return indexes[i - w:i + w + 1]
        elif i - w >= 0 and i + w + 1 > n:
            indexes_ret = indexes[i - w:n]
            # remain 2*w+1-(n-i+w)
            indexes_ret.extend(indexes[0:(2 * w + 1 - (n - i + w))])
        elif i - w < 0 and i + w + 1 <= n:
            indexes_ret = indexes[0:i + w + 1]
            indexes_ret.extend(indexes[n - (w - i):n])
        assert len(indexes_ret) == 2 * w + 1
        return indexes_ret

    def filter_indexes(self, indexes, w=1):
        '''
        使用类似中值滤波的技术，对轮廓进行去噪
        @param indexes: 轮廓
        @param w:       窗口的大小
        '''
        assert isinstance(indexes, list)
        indexes_ret = []
        n = len(indexes)
        for i in xrange(n):
            indexes_ret.append(
                np.median(self.get_indexes_in_window(indexes, i, w))
            )
        return indexes_ret

    def rename_indexes(self, indexes):
        '''@note 怎么处理具体应用和函数之间的矛盾呢？'''
        '''可以假设最开始的一个index是属于左上角的点么, 额额。'''
        assert isinstance(indexes, list)
        name_map = {}
        c = 0
        indexes_ret = []
        for i in indexes:
            if i not in name_map:
                name_map[i] = c
                c = c + 1
            indexes_ret.append(name_map[i])
        return indexes_ret

    def adjust_indexes(self, indexes, w=1):
        '''
        对contour中的点进行整理，包括两方面：
            1. 理想情况下，各个标号应该是连续的，因为各边应当也是连续的，对于不连续的噪声点，应当予以抛弃。
               抛弃了之后，每条边里面的点也应当大于等于2
            2. 为了便于后期处理，标号进行重命名，保证相邻的标号是连续变化的，即[0][0][0][1][1][1][2][2][2]...
        @param indexes: 点的list，表示一个
        @param idx:     每个点所属的直线index, 是一个包含整形的list
        @param n:       插值进行的轮数
        '''
        assert isinstance(indexes, list)
        # interpolate
        # check and median filtering
        # rename indexes
        indexes_ret = self.filter_indexes(indexes, w)
        indexes_ret = self.rename_indexes(indexes_ret)
        return indexes_ret

    def fit_lines_from_points(self, points, idx):
        '''
        @param points: list of points
        @param idx:    index of which line each point belongs to
        '''
        # store line results
        lines = []
        # number of lines
        line_count = max(idx) + 1
        # init 2d point array
        list_line_points = [[] for i in xrange(line_count)]
        for i in xrange(len(points)):
            # put point in corresponding lines
            list_line_points[idx[i]].append(points[i])
        for i in xrange(line_count):
            # fit line from points
            line = cv2.fitLine(np.asarray(list_line_points[i],
                                          dtype=np.float32),
                               cv2.cv.CV_DIST_L2,
                               0, 0.01, 0.01)
            lines.append(line)
        return lines

    def get_sorted_corners(self, lines, w, h):
        '''
        @deprecated: 一般不需要对定点进行排序，如果真的有需要，请使用sort_corners
        @param lines:
        @param w:
        @param h:
        '''
        raise DeprecationWarning()
        return

    def get_vertices(self, lines):
        """要求直线是首尾相接的，直接计算相邻直线的交点得到顶点"""
        # init point list
        corners = []
        for i in xrange(len(lines)):
            # get intersection of lines
            corner = self.get_intersection_point(lines[i],
                                                 lines[(i + 1) % len(lines)])
            # add point to corners
            corners.append(corner)
        corners = np.asarray(corners)
        return corners

    def is_point_inbound(self, p, w, h):
        if (p[0] >= 0 and p[0] < w) and (p[1] >= 0 and p[1] < h):  # FIXME h,w?
            return True
        else:
            return False

    def sort_corners(self, corners):
        '''
        @note  使用Graham Scan可以完成Sort
        @param corners: list of points to be sorted
        '''
        raise Exception('Not Implemented')

    def get_intersection_point(self, l, r):
        '''
        @param l: line in format (vx, vy, x0, y0)
        @param r: line in format (vx, vy, x0, y0)
        @summary: get the intersection of l and r by solving linear equation
        '''
        l = l.flatten()
        r = r.flatten()
        a = np.array([[l[1], -l[0]], [r[1], -r[0]]])
        b = np.array([[l[2] * l[1] - l[3] * l[0]],
                      [r[2] * r[1] - r[3] * r[0]]])
        point = np.linalg.solve(a, b)

        return point


def get_sub_image(img, start_point, end_point):
    sub_image = img[start_point[1]:end_point[1] + 1,
                    start_point[0]:end_point[0] + 1]
    return sub_image


def main():
    # Create a black image, a window and bind *draw_rectangle* to it
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', RectangleDrawer.draw_rectangle)

    # Read image
    GlobalVariable.original_image = cv2.imread('img/corridor.jpg')
    GlobalVariable.original_image = cv2.resize(GlobalVariable.original_image,
                                               (0, 0), fx=0.3, fy=0.3)
    if GlobalVariable.original_image is None:
        logging.error('Image Not Found')
        assert 0
    # image for display
    GlobalVariable.display_image = copy.deepcopy(GlobalVariable.original_image)

    # Show image
    cv2.imshow("image", GlobalVariable.display_image)

    # Wait for user input,
    # the target rectangle should be drawn before hitting carret
    key = cv2.waitKey(0)

    # if carret is hit
    if key == ord('\r'):
        # extract target region
        sub_image = get_sub_image(GlobalVariable.original_image,
                                  GlobalVariable.start_point,
                                  GlobalVariable.end_point)
        # transform image
        transformer = ImageTransformer(sub_image)
        transformed_image = transformer.transform()
        # show result and wait
        cv2.imshow("transformed", transformed_image)
        cv2.waitKey(0)

    # clean up windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print 'Hello'
    main()
