#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2014-5-23

@author: zx
'''
import copy
import cv2
import numpy as np

import logging


class Plotter(object):

    @staticmethod
    def __convert_line_format(l, s):
        '''
        @summary: Convert line from format (vx, vy, x0, y0) to (x0, y0, x1, y1)
        @return:  Converted line in format (x0, y0, x1, y1)
        @param    l: input line in format (vx, vy, x0, y0)
        @param    s: size of image
        '''
        h = s[0]
        w = s[1]
        l = copy.copy(l)
        l[0] = l[0] * 20 + l[2]
        l[1] = l[1] * 20 + l[3]
        if l[0] >= w or l[0] < 0 or l[1] >= h or l[0] < 0:
            raise Exception('Line Format Conversion Failed')
        else:
            return l

    @staticmethod
    def plot_lines(image, lines, title='lines'):
        '''
        @summary 本函数用于Debug时，展示提取出的直线在原图中的的位置
        @param   image: target image
        @param   lines: list of lines, in format (vx, vy, x0, y0)
        '''
        logging.info("Enter plot_lines")
        tmp = copy.deepcopy(image)
        lines = copy.deepcopy(lines)
        s = image.shape[0:2]
        for i in xrange(len(lines)):
            l = lines[i]
            l = Plotter.__convert_line_format(l, s)
            cv2.line(tmp, (l[0], l[1]), (l[2], l[3]), (255, 255, 0), 2)
            cv2.putText(tmp, 'l%d' % i, (l[0] - 20, l[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
        cv2.imshow(title, tmp)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
        logging.info("Leave plot_lines")

    @staticmethod
    def plot_points(image, points, title='points'):
        color = (255, 255, 0)
        tmp = copy.copy(image)
        # circle require points to be int type
        points = np.asarray(points, dtype=np.int)
        for i in xrange(len(points)):
            p = points[i]
            cv2.circle(tmp, (p[0], p[1]), 2, color)
            cv2.imshow(title, tmp)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    @staticmethod
    def plot_image(image, title):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    @staticmethod
    def plot_contours(image, contours, title='contours'):
        contours = np.array(contours, np.int)
        tmp = copy.copy(image)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        tmp *= 255
        cv2.drawContours(tmp, contours, 0, (0, 0, 255), 2)
        cv2.imshow(title, tmp)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
