#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import copy

import cv2  # OpenCV 2
import numpy as np

from research.util import get_elements_in_window, distance_from_point_to_line,\
                          info
from research.corner.global_variable import GlobalVariable


class ContourAnalyzer(object):

    def fit_lines_from_points(self, points, idx):
        '''
        @param points: list of points
        @param idx:    index of which line each point belongs to

        @return: lines computed from points according to idx
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
                np.median(get_elements_in_window(indexes, i, w))
            )
        return indexes_ret

    def rename_indexes(self, indexes):
        '''
        将indexes中的元素重命名，
        例如[2, 2, 3, 3, 1, 2]会重命名为[0, 0, 1, 1, 2, 0]
        @param indexes:
        @return: renamed indexes
        '''
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


class ContourAnalyzerRANSAC(ContourAnalyzer):

    def remove_unused_points(self, contour, idx):
        assert max(idx) != -1
        list_of_remove = []
        idx_new = []
        for k, v in enumerate(idx):
            if v < 0:  # remove -1 and -2
                list_of_remove.append(k)
            else:
                idx_new.append(v)
        contour = np.delete(contour, list_of_remove, 0)
        return contour, idx_new

    def extract_lines(self, contour, k):
        """extract lines from contour"""
        idx = self.get_idx_from_contours_ransac(contour, k)
        info('best idx', idx, '\n')
        info('idx count', len(idx), '\n')
        # prepare points for line fitting
        contour, idx = self.remove_unused_points(contour, idx)
        idx = self.rename_indexes(idx)
        # fit points to line
        lines = self.fit_lines_from_points(contour, idx)
        return lines

    def compute_inliner_rate(self, inliner_idx):
        n = len(inliner_idx)
        inliner = 0
        for i in inliner_idx:
            inliner = inliner + 1 if i >= 0 else inliner
        return inliner / n

    # TODO: determine N
    # TODO: determine d
    def get_idx_from_contours_ransac(self, contour, k):
        # N = log(1-p)/log(1-(1-e)^s)
        N = 100
        # d = \sqrt {3.84 * \sigma ^ 2}
        d = 2.5

        best_idx = []
        best_rate = -1
        for _ in xrange(N):
            # do RANSAC for one pass
            try:
                inliner_idx = self.one_pass_ransac(contour, d, k)
            except Exception:
                continue
            inliner_rate = self.compute_inliner_rate(inliner_idx)
            if inliner_rate > best_rate:
                best_rate = inliner_rate
                best_idx = inliner_idx
                info('best_rate updated to:', best_rate, '\n')
        return best_idx

    def one_pass_ransac(self, contour, d, k):
        n = len(contour)  # points left
        inliner_idx = [-1 for _ in xrange(n)]  # -1 means not included yet
        for _ in xrange(k):
            p = np.random.randint(n)  # choose one as the central point
            inliner_idx = self.delete_one_edge(contour, inliner_idx, d, p)
            n = len(filter(lambda x: x == -1, inliner_idx))
        return inliner_idx

    def set_index_by_step(self, contour, inliner_idx, idx,
                          step, p, n, a, delta):
        length = len(contour)
        while True:
            q = p % length
            d = distance_from_point_to_line(n, a, contour[q])
            if d < delta:
                if inliner_idx[q] == -1:
                    inliner_idx[q] = idx
                else:  # remove index in common region
                    inliner_idx[q] = -2  # DO NOT USE -1
            else:
                break
            p = p + step

    # Add Test
    def set_adjacent_inliner(self, contour, inliner_idx, p, n, a, delta=2):
        idx = max(inliner_idx) + 1
        # delete forward
        self.set_index_by_step(contour, inliner_idx, idx,
                               1, p, n, a, delta)
        # delete backward
        self.set_index_by_step(contour, inliner_idx, idx,
                               -1, p - 1, n, a, delta)
        return inliner_idx

    def delete_one_edge(self, contour, inliner_idx, d, p):
        '''
        从contour中删除和第p个未被分配的点在同一条直线上的点，点到直线的最大距离为d
        @param contour: 轮廓
        @param inliner_idx: 当前点已经完成的分配
        @param d: 最大偏差
        @param p: 中心点的位置
        '''
        assert isinstance(contour, np.ndarray)
        counter = 0
        center = 0  # when the contour is shorter than p, center will be 0
        for k, v in enumerate(inliner_idx):
            counter = counter + 1 if v == -1 else counter
            if counter == p:
                center = k
                break
        # TODO: determine Region of Support (ROS)
        points = get_elements_in_window(contour, center, 2)
        # get line in format (vx, vy, x0, y0)
        line = cv2.fitLine(np.asarray(points,
                                      dtype=np.float32),
                                      cv2.cv.CV_DIST_L2,
                                      0, 0.01, 0.01)
        vx = line[0]
        vy = line[1]
        n = np.asarray([vx, vy])
        n = np.squeeze(n)
        a = np.asarray([line[2], line[3]])
        a = np.squeeze(a)
        self.set_adjacent_inliner(contour, inliner_idx, center, n, a, d)
        return inliner_idx


class ContourAnalyzerClustering(ContourAnalyzer):

    def extract_lines(self, contour, n):
        """extract lines from contour"""
        # get convex contour of a contour
        contour = np.squeeze(cv2.convexHull(contour))
        # assign points to different lines
        idx = self.get_idx_from_contours_kmeans(contour, n)
        # prepare points for line fitting
        idx = self.adjust_indexes(idx)
        # fit points to line
        lines = self.fit_lines_from_points(contour, idx)
        return lines

    def get_idx_from_contours_kmeans(self, contour, n):
        '''
        @param contour: contour in points, assumed to be convex
        @param n: number of sides
        '''
        from research.util import Plotter

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

    def get_idx_from_contours_spectral_clustering(self, contours):
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


class CornerExtractor(object):
    def __init__(self, image):
        self.image = copy.deepcopy(image)

    def change_points_to_nearest(self, contour, points):
        from research.util import Plotter
        Plotter.plot_points(GlobalVariable.original_image, contour,
                            'Points of original')
        contour = np.squeeze(contour)
        points = np.squeeze(points)
        result = []
        for p1 in points:
            p3 = min(contour,
                key=lambda p2:
                    np.linalg.norm(np.asarray(p1, np.float) -
                                   np.asarray(p2, np.float)))
            result.append(p3)
        return result

    def extract(self, n, convex=True):
        return self.get_bounding_polygon_vertices(self.image, convex, n)

    def get_bounding_polygon_vertices(self, image, convex, n):
        from research.util import Plotter

        lines, contour = self.get_lines(image, convex, n)
        Plotter.plot_lines(image, lines, 'final results')
        corners = self.get_vertices(lines)
        corners = self.change_points_to_nearest(contour, corners)
        return corners

    def get_contour_from_image(self, image):
        from research.util import Plotter

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
                                      cv2.CHAIN_APPROX_NONE)
#                                       cv2.CHAIN_APPROX_SIMPLE)
#                                       cv2.CHAIN_APPROX_TC89_L1)
        # get the contour with the most points
        contour = reduce(lambda x, y: x if len(x) > len(y) else y,
                         contours, [])
        Plotter.plot_image(bw_img, 'bw_img_after_contour')
        Plotter.plot_contours(bw_img, [contour], 'extracted contour')
        contour = np.vstack(contour).squeeze()
        # plot contour
        Plotter.plot_points(bw_img, contour,
                            'the points of contour after squeezing')
        # interpolate
#         contour = self.interpolate_points(contour)
        # plot contour
#         Plotter.plot_points(bw_img, contour)
        Plotter.plot_contours(bw_img, [contour], 'contour after squeezing')
        return contour

    def get_lines(self, image, convex, n):
        '''
        @summary: extract boundary lines of polygon from image
        @param image: input image with polygon

        @return: extracted lines
        '''
        contour = self.get_contour_from_image(image)
        if convex:
            analyzer = ContourAnalyzerClustering()
        else:
            analyzer = ContourAnalyzerRANSAC()
        lines = analyzer.extract_lines(contour, n)
        return lines, contour

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
        @note  使用Graham Scan可以完成Sort, 从最低点开始扫描，测量点和矫正点可以一一对应
        @param corners: list of points to be sorted
        '''
        raise Exception('Not Implemented')

    def get_intersection_point(self, l, r):
        '''
        get the intersection of l and r by solving linear equation

        @param l: line in format (vx, vy, x0, y0)
        @param r: line in format (vx, vy, x0, y0)
        '''
        l = l.flatten()
        r = r.flatten()
        a = np.array([[l[1], -l[0]], [r[1], -r[0]]])
        b = np.array([[l[2] * l[1] - l[3] * l[0]],
                      [r[2] * r[1] - r[3] * r[0]]])
        # TODO: 可能出现Singular Matrix: raise LinAlgError, 'Singular matrix'
        point = np.linalg.solve(a, b)

        return point


def get_sub_image(img, start_point, end_point):
    sub_image = img[start_point[1]:end_point[1] + 1,
                    start_point[0]:end_point[0] + 1]
    return sub_image
