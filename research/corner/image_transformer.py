#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import cv2
import numpy as np


class ImageTransformer:
    def __init__(self, image):
        # get a copy of input image
        self.image = copy.deepcopy(image)

    def align_points(self, points_original):
        '''
        调整点的位置，取离远点最近的点作为第一个点
        @return: 调整之后的点的list
        '''
        p, _ = min(enumerate(points_original),
                   key=lambda (k, v): np.linalg.norm(v))
        n = len(points_original)
        points_original_new = [points_original[(i + p) % n] for i in xrange(n)]
        return points_original_new

    def transform(self, n=4):
        '''
        @summary: 这个方法会从image中提取顶点，然后通过仿射变换将其转化为标准的大小
        '''
        from research.corner.corner_extraction import CornerExtractor
        from research.util import Plotter

        # extract corners in image
        extractor = CornerExtractor(self.image)
        points_original = extractor.extract(n)
        # get the corresponding target points
        points_mapped = self.get_mapped_points()
        points_original = self.align_points(points_original)
        Plotter.plot_points(self.image, points_mapped, "mapped points")
        Plotter.plot_points(self.image, points_original, "original points")
        # compute and apply perspective transform
        points_original = np.asarray(points_original,
                                     'float32'
                                    ).reshape((4, 2))
        points_mapped = np.asarray(points_mapped,
                                   'float32'
                                  ).reshape((4, 2))
        H = cv2.getPerspectiveTransform(points_original, points_mapped)
        transformed_image = cv2.warpPerspective(self.image, H, (300, 300))

        return transformed_image

    def get_mapped_points(self, n=4):
        '''
        @summary: 给出了对应的标准点。默认为矩形，其它情况则自动生成。
                  矩形:     使用getPerspectiveTransform得到放射变换
                  超过四点:  使用findHomography得到对应的放射变换
        @note:    根据具体应用的需要，由使用者设定相应的点，第一个点总是[0, 0],
        @fix:     顺时针还是逆时针，还得结合contour的返回结果
        '''
        if n <= 3:
            raise Exception("Invalid Vertex Number")

        if n == 4:
            # set corresponding points
            w = 60
            h = 100
            corners = [[0, 0], [w, 0], [w, h], [0, h]]
        else:
            assert False, 'Target Points Not Found'
        return corners
