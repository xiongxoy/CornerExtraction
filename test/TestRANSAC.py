#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2014-6-28

@author: Zhou Xiong
'''
import unittest
import numpy as np

from research.util import line, distance_from_point_to_line
from research.corner.corner_extraction import ContourAnalyzerRANSAC
from research.corner.global_variable import GlobalVariable


class Test(unittest.TestCase):

    def setUp(self):
        GlobalVariable.original_image = np.zeros((200, 200), np.float)
        self.analyzer = ContourAnalyzerRANSAC()

    # FIXME: 直线没有被正确的提取
    def test_ransac_on_rectangle(self):
        print GlobalVariable.original_image
        rect = line(10, 10, 10, 100)
        rect.extend(line(100, 10, 10, 10))
        rect = np.asarray(rect, np.int)
        print self.analyzer.extract_lines(rect, 2)

    def test_distance_from_line(self):
        '''
        测试点到直线距离计算的准确性
        '''
        n = np.asarray([0, 10], np.float)
        a = np.asarray([0, 0], np.float)
        point = np.asarray([1.5, 10], np.float)
        d = distance_from_point_to_line(n, a, point)
        np.testing.assert_almost_equal(d, 1.5)

    def test_delete_one_edge(self):
        '''
        此函数测试RANSAC的核心部分，需要从contour中删除和指定点
        在同一直线上的点
        '''
        pass

    def test_set_adjacent_inliner(self):
        '''
        测试指定点周围的点是否被恰当的设定了idx
        '''
        points = line(0, 0, 0, 100)
        points.extend(line(10, 0, 10, 100))
        inliner_idx = [-1] * len(points)
        n = np.asarray([0, 10], np.float)
        a = np.asarray([0, 0], np.float)
        inliner_idx = self.analyzer.set_adjacent_inliner(points,
                                                         inliner_idx, 3, n, a)
        print len(points), points
        print len(inliner_idx), inliner_idx
        inliner_idx = filter(lambda x: x != -1, inliner_idx)
        self.assertEqual(len(inliner_idx), len(points) / 2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
