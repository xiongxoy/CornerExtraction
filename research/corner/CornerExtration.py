#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division      # 1/2 = 0.5
import cv2                           # OpenCV 2
import numpy as np

import math
import copy

import logging

#===============================================================================
# Global Variables
#===============================================================================
class GlobalVariable:
    start_point = (0, 0)
    end_point = (0, 0)
    display_image = []
    original_image = []
    state = 0

#===============================================================================
# Drawing Rectangle
#===============================================================================
# mouse callback function
class RectagleDrawer:  
    @staticmethod
    def draw_rectangle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: 
            if GlobalVariable.state == 0:
                GlobalVariable.state = 1
                RectagleDrawer.start_rectangle(x, y)
            elif GlobalVariable.state == 1:
                GlobalVariable.state = 2
        elif event == cv2.EVENT_MOUSEMOVE:
            if GlobalVariable.state == 1:
                RectagleDrawer.update_rectangle(x, y)
            if GlobalVariable.state != 2:
                RectagleDrawer.draw_rectangle_on_image()
    @staticmethod
    def start_rectangle(x, y):
        GlobalVariable.start_point = (x, y)
        GlobalVariable.end_point = (x, y)
    @staticmethod
    def update_rectangle(x, y):
        GlobalVariable.end_point = (x, y)
    @staticmethod
    def draw_rectangle_on_image():
        GlobalVariable.display_image = copy.deepcopy(GlobalVariable.original_image)
        cv2.rectangle(GlobalVariable.display_image, GlobalVariable.start_point, GlobalVariable.end_point, (0, 0, 255))
        cv2.imshow("image", GlobalVariable.display_image)

class Plotter:
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
        l[0] = l[0] * 20 + l[2]
        l[1] = l[1] * 20 + l[3]
        if l[0] >= w or l[0] < 0 or l[1] >= h or l[0] < 0:
            raise Exception('Extension Failed')
        else:
            return l
    @staticmethod
    def plot_lines(image, lines):
        '''
        @summary 本函数用于Debug时，展示提取出的直线在原图中的的位置
        @param   image: target image
        @param   lines: list of lines, in format (vx, vy, x0, y0)
        '''
        tmp = copy.deepcopy(image)
        s = image.shape[0:2]
        for i in xrange(len(lines)):
            l = lines[i]
            l = Plotter.__convert_line_format(l, s)
            cv2.line(tmp, (l[0], l[1]), (l[2], l[3]), (255, 255, 0), 2)
            cv2.putText(tmp, 'l%d' % i, (l[0] - 20, l[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
        cv2.imshow('plot lines', tmp)
        cv2.waitKey(0)
        cv2.destroyWindow('plot lines')
    @staticmethod
    def plot_points(image, points):
        color = (255, 255, 0)
        print points, len(points)
        tmp = copy.copy(image)
        for i in xrange(len(points)):
            p = points[i]
            cv2.circle(tmp, (p[0], p[1]), 2, color)
            cv2.imshow('points', tmp)
        cv2.waitKey(0)
        cv2.destroyWindow('points')   
    @staticmethod
    def plot_image(image, title):
        cv2.imshow(title, image)
        cv2.waitKey(0) 
        cv2.destroyWindow(title)
    @staticmethod
    def plot_contours(image, contours):
        tmp = copy.copy(image)
        cv2.drawContours(tmp, contours, 0, (0, 0, 255), 2)
        cv2.imshow('contours', tmp)
class CornerExtractor:
    def __init__(self, image):
        self.image = copy.deepcopy(image)
    def extract(self, n=4, convex=True):
        return  self.get_bounding_polygon_vertices(self.image)

    def get_bounding_polygon_vertices(self, image):
        lines = self.get_lines(image)
        corners = self.get_vertices(self, lines)
        return corners
    def get_bounding_rect_vertices(self, image):
        '''
        single-purpose function to extract vertex  
        @param image: image of rectangle 
        '''
        lines = self.get_lines(image)
        h, w = image.shape[0:2]
        corners = self.get_vertices(lines, w, h)
        return corners

    def get_idx_from_contours_convex(self, contours):
        # cv2.drawContours(image, contours[0], 0,(0,0,255), 2)
        hull = cv2.convexHull(contours)
        # cv2.drawContours(image,[hull],0,(0,0,255),2)
        k = []
        for i in range(hull.shape[0]):
            # cv2.circle(image, tuple(hull[i][0]), 2, (255,255,0))
            ki = hull[ (i + 1) % hull.shape[0] ] - hull[i]
            ki = ki[0]
            ki = ki.astype(float)
            ki_mod = (ki[0] ** 2 + ki[1] ** 2) ** 0.5
            ki[0] = ki[0] / ki_mod
            ki[1] = ki[1] / ki_mod
            k.append(ki)
    
        termination_criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        k_array = np.float32(k)
        centers = cv2.kmeans(k_array, 4, termination_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        idx = centers[1]
        return hull, idx
    def get_idx_from_contours_concave(self, contours):
        cv2.drawContours(self.image, [contours], 0, (0, 0, 255), 2)
        k = []
        for i in range(contours.shape[0]):
            # cv2.circle(image, tuple(contours[i][0]), 2, (255,255,0))
            ki = contours[ (i + 1) % contours.shape[0] ] - contours[i]
            ki = ki[0]
            ki = ki.astype(float)
            ki_mod = (ki[0] ** 2 + ki[1] ** 2) ** 0.5
            ki[0] = ki[0] / ki_mod
            ki[1] = ki[1] / ki_mod
            ki.append(contours[0], contours[1])
            k.append(ki)
     
        termination_criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
        k_array = np.float32(k)
        centers = cv2.kmeans(k_array, 4, termination_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        idx = centers[1]
        return contours, idx

        # build an array of direction+position -> what are the possible better choice?
        # get the index of each point
            # cluster the array, spectral clustering is a good choice
            # find lines in the 4d space
        # smooth the result using a 1*5 median filter
        raise Exception('To be implemented')
    def get_lines(self, image):
        '''
        @summary: extract boundary lines of polygon from image
        @param image: input image with polygon
        
        @return: extracted lines 
        '''
        # detect all edges in the image
        bw_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to gray image
        bw_img = cv2.blur(bw_img, (3, 3))                 # blur with mean value filter
        bw_img = cv2.Canny(bw_img, 50, 150)               # detect edges, sensitive to the parameter of canny
        cv2.imshow('bw_img', bw_img)                      # show result
   
        # find contour of the edges  
        contours, __ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # get contour
        Plotter.plot_points(bw_img, np.vstack(contours).squeeze())                           # plot contour
        
        # get lines from contour 
        contours, idx = self.get_idx_from_contours_concave(contours)  # assign points to different lines
        lines = self.fit_lines_from_points(contours, idx)             # fit points to line

        return lines
    def fit_lines_from_points(self, points, idx):
        '''
        @param points: list of points
        @param idx:    index of which line each point belongs to
        '''
        lines = []                                              # store line results
        line_count = max(idx) + 1                               # number of lines
        list_line_points = [ [] for i in xrange(line_count) ]   # init 2d point array
        for i in xrange(len(points)):
            list_line_points[idx[i]].append(points[i])          # put point in corresponding lines
        for i in xrange(line_count):
            line = cv2.fitLine(np.array(list_line_points[i]), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)     # fit line from points
            lines.append(line)
        return lines
    def get_sorted_corners(self, lines, w, h):
        '''
        @deprecated: 一般不需要对定点进行排序，如果真的有需要，请使用sort_corners 
        @param lines:
        @param w:
        @param h:
        '''
        corners = self.get_vertices(lines, w, h)
        corners = [corners[0], corners[2], corners[3], corners[1]]
        return corners
    def get_vertices(self, lines, w, h):
        """要求直线是首尾相接的，直接计算相邻直线的交点得到顶点"""
        corners = []                                                                     # init point list 
        for i in xrange(len(lines)):
            corner = self.get_intersection_point(lines[i], lines[(i + 1) % len(lines)])  # get intersection of lines
            corners.append(corner)                                                       # add point to corners
        corners = np.asarray(corners)                                                    
        return corners
    def is_point_inbound(self, p, w, h):
        if (p[0] >= 0 and p[0] < w) and (p[1] >= 0 and p[1] < h):  # FIXME h,w?
            return True
        else:
            return False
    def sort_corners(self, corners):
        '''
        :todo  使用Graham Scan可以完成Sort
        :param corners: list of points to be sorted
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
        a = np.array([ [l[1], -l[0]]         , [r[1], -r[0]] ])
        b = np.array([ [l[2] * l[1] - l[3] * l[0]] , [r[2] * r[1] - r[3] * r[0]] ])
        point = np.linalg.solve(a, b)
    
        return point
#===============================================================================
#Transforming Image
#===============================================================================
class ImageTransformer:
    def __init__(self, image):
        # get a copy of input image 
        self.image = copy.deepcopy(image)
    def transform(self, n=4):
        '''
        @summary: 这个方法会从image中提取顶点，然后通过仿射变换将其转化为标准的大小
        '''
        # extract corners in image
        extractor = CornerExtractor(self.image)
        points_original = extractor.extract()
        # get the corresponding target points 
        points_mapped   = self.get_mapped_points()

        # convert storage formats of points
        points_original = np.asarray(points_original, 'float32').reshape((4, 2))
        points_mapped   = np.asarray(points_mapped, 'float32').reshape((4, 2))
        # apply perspective transform
        H = cv2.getPerspectiveTransform(points_original, points_mapped)
        transformed_image = cv2.warpPerspective(self.image, H, (300, 300))

        return transformed_image
    def get_mapped_points(self, n=4):
        '''
        @summary: 给出了对应的标准点。默认为矩形，其它情况则自动生成。
                  矩形:     使用getPerspectiveTransform得到放射变换
                  超过四点:  使用findHomography得到对应的放射变换
        @note:    由于多点对应有困难，可以考虑取其bounding rectangle，
                  然后通过bounding rectangle的映射来做还原
        '''
        if n <= 2:
            raise Exception("Invalid Vertex Number")

        if n == 4: 
            # set corresponding points
            w = 60
            h = 100
            corners = [[0, 0], [w, 0], [w, h], [0, h]]
        else:
            # generate clock-wise points 
            r = 100                     # radius
            deg = math.pi               # initial position, i.e. start from top point
            delta = math.pi * 2 / n      
            corners = [ [r*math.cos(deg-i*delta), r*math.sin(deg-i*delta)] for i in xrange(n) ]

        return corners

def get_sub_image(img, start_point, end_point):
    sub_image = img[start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1]
    return sub_image
  
def main():
    global original_image, display_image

    # Create a black image, a window and bind *draw_rectangle* to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)
   
    # Read image  
    original_image = cv2.imread('img/concave-polygon.png')  # original image
    display_image = copy.deepcopy(original_image)           # image for display
    
    # Show image
    cv2.imshow("image", display_image)
    
    # Wait for user input, the target rectangle should be drawn before hitting carret
    key = cv2.waitKey(0)
    
    # if carret is hit 
    if key == ord('\r'):
        # extract target region
        sub_image = get_sub_image(original_image, start_point, end_point)
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
