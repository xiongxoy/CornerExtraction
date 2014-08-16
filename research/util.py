#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2014-5-23

@author: Zhou Xiong
'''
import copy
import logging

import cv2
import numpy as np

freeman_code = np.asarray([[1, 0], [1, -1], [0, -1], [-1, -1],
                           [-1, 0], [-1, 1], [0, 1], [1, 1]], np.float)


#TODO: 实现freeman chain code的可视化
def convert_chiancode(code, n=1):
    '''
    根据制定的大小，将freeman chain code转化为可视化的图形
    @param code: 图形的freeman chain code
    @param n: 每次的步长
    @return: contour represented by code
    '''
    size = 300
    point = np.array([size / 2, size / 2], dtype=np.float)
    points = []
    tmp = point
    points.append(tmp)
    for c in code:
        for _ in xrange(n):
            tmp = tmp + freeman_code[int(c)]
            points.append(tmp)
    return points


def info(*message, **dict_args):
    for e in message:
        print e,


def add_noise(x, y):
    noise = np.random.normal(0, 2, 2)
    x = x + noise[0]
    y = y + noise[1]
    return (x, y)


def line(x0, y0, x1, y1):
    "Bresenham's line algorithm"
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points


def line_noisy(x0, y0, x1, y1):
    "Bresenham's line algorithm, with noise"
    points = line(x0, y0, x1, y1)
    points = [add_noise(p) for p in points]
    return points


# http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
def distance_from_point_to_line(n, a, point):
    '''
    计算从point到方向向量为n，通过a点的直线的距离
    @param n:
    @param a:
    @param point:
    '''
    n = n / np.linalg.norm(n)
    dist_vec = (a - point) - (np.inner((a - point), n)) * n
    return np.linalg.norm(dist_vec)


def get_elements_in_window(li, i, w):
    li = list(li)
    n = len(li)
    elements_ret = []
    if i - w >= 0 and i + w + 1 <= n:
        return li[i - w:i + w + 1]
    elif i - w >= 0 and i + w + 1 > n:
        elements_ret = li[i - w:n]
        # remain 2*w+1-(n-i+w)
        elements_ret.extend(li[0:(2 * w + 1 - (n - i + w))])
    elif i - w < 0 and i + w + 1 <= n:
        elements_ret = li[0:i + w + 1]
        elements_ret.extend(li[n - (w - i):n])
    assert len(elements_ret) == 2 * w + 1
    return elements_ret


def interpolate_points(contour, n=1):
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
            cv2.putText(tmp, 'L%d' % i, (l[0] - 20, l[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        cv2.imshow(title, tmp)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
        logging.info("Leave plot_lines")

    @staticmethod
    def plot_points(image, points, title='points'):
        color = (0, 0, 255)
        tmp = copy.copy(image)
        try:
            tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        except cv2.error as e:
            print 'error is: ', e
            pass
        #  circle require points to be int type
        points = np.array(points, dtype=np.int)
        for i in xrange(len(points)):
            p = points[i]
            cv2.circle(tmp, (p[0], p[1]), 2, color, 2)
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
