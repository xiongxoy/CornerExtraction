#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2014-6-28

@author: Zhou Xiong
'''
import unittest
import numpy as np

from research.util import line, distance_from_point_to_line, Plotter
from research.corner.corner_extraction import ContourAnalyzerRANSAC,\
    CornerExtractor
from research.corner.global_variable import GlobalVariable


def get_point_by_idx(points, inliner_idx, idx):
    '''
    从点的列表中提取index为指定index的点
    @param points: 点的list
    @param inliner_idx: 对list中点的标记
    @param idx: 目标index
    '''
    result = []
    for k, v in enumerate(points):
        if inliner_idx[k] == idx:
            result.append(v)
    return result


class Test(unittest.TestCase):

    def setUp(self):
        GlobalVariable.original_image = np.zeros((200, 200), np.float)
        self.analyzer = ContourAnalyzerRANSAC()

    @unittest.skip('')
    def test_ransac_on_rectangle(self):
        '''
        测试RANSAC直线提取在两条相邻的直线上的效果
        '''
        rect = line(10, 10, 10, 100)
        rect.extend(line(100, 10, 10, 10))
        rect = np.asarray(rect, np.int)
        lines = self.analyzer.extract_lines(rect, 2)
        Plotter.plot_lines(GlobalVariable.original_image,
                           lines, 'show fitted line')

    @unittest.skip('')
    def test_distance_from_line(self):
        '''
        测试点到直线距离计算的准确性
        '''
        n = np.asarray([0, 10], np.float)
        a = np.asarray([0, 0], np.float)
        point = np.asarray([1.5, 10], np.float)
        d = distance_from_point_to_line(n, a, point)
        np.testing.assert_almost_equal(d, 1.5)

    @unittest.skip('')
    def test_distance_from_diagnal_line(self):
        rect = line(10, 10, 100, 100)   # (10, 10)到(100, 100)的斜线
        # line in format (vx, vy, x0, y0)
        line_param = [0.70710677, 0.70710677, 13., 13.]
        n = np.asarray(line_param[0:2], np.float)
        a = np.asarray(line_param[2:], np.float)
        for p in rect:
            d = distance_from_point_to_line(n, a, p)
            print 'point:', p
            print d

    @unittest.skip('')
    def test_delete_one_edge_1(self):
        '''
        此函数测试RANSAC的核心部分，需要从contour中删除和指定点
        在同一直线上的点
        '''
        rect = line(10, 10, 10, 100)
        rect.extend(line(100, 10, 10, 10))
        rect = np.asarray(rect, np.int)
        n = len(rect)
        inliner_idx = [-1 for _ in xrange(n)]
        d = 1.5
        p = 4
        inliner_idx = self.analyzer.delete_one_edge(rect,
                                                    inliner_idx, d, p)
        inliner_idx = filter(lambda x: x != 0, inliner_idx)
        first = len(inliner_idx)
        second = len(rect) / 2
        delta = second / 10
        self.assertAlmostEqual(first, second, delta=delta)

    @unittest.skip('')
    def test_delete_one_edge_2(self):
        '''
        在实验的时候效果不好，怀疑是删除斜线的时候，有问题
        这个单元测试来检查斜线是否和预期一样被删除
        @update: 修改了点到直线距离的计算函数，现在可以了
        '''
        # prepare data
        rect = line(10, 10, 100, 100)   # (10, 10)到(100, 100)的斜线
        rect.extend(line(100, 10, 10, 10))  # (10, 10)到(100, 10)的直线
        rect = np.asarray(rect, np.int)
        Plotter.plot_points(GlobalVariable.original_image,
                            rect, 'image of diagonal line')
        # delete edge
        n = len(rect)
        inliner_idx = [-1 for _ in xrange(n)]
        d = 1.5
        p = 4
        inliner_idx = self.analyzer.delete_one_edge(rect,
                                                    inliner_idx, d, p)
        print 'a', inliner_idx
        # show result
        rect_new = get_point_by_idx(rect, inliner_idx, 0)
        self.assertTrue(len(rect_new) != 0, 'Error: 没有找到对应点')
        Plotter.plot_points(GlobalVariable.original_image,
                            rect_new, 'image after deletion')
        # test length
        inliner_idx = filter(lambda x: x != 0, inliner_idx)
        first = len(inliner_idx)
        second = len(rect) - len(line(10, 10, 100, 100))
        delta = second / 10
        self.assertAlmostEqual(first, second, delta=delta)

    @unittest.skip('')
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
        inliner_idx = filter(lambda x: x != -1, inliner_idx)
        self.assertEqual(len(inliner_idx), len(points) / 2)

    def test_change_points_to_nearest(self):
        '''
        测试将点替换成contour与其最邻近的算法
        @note: 测试通过
        '''
        extractor = CornerExtractor(None)
        point_set = line(0, 0, 100, 100)
        points = [[1, 1], [50, 50], [0, 100]]
        actual = extractor.change_points_to_nearest(point_set, points)
        expteced = [[1, 1], [50, 50], [50, 50]]
        print actual, expteced
        np.testing.assert_array_almost_equal(actual, expteced)

    @unittest.skip('')
    def test_skip(self):
        self.assertTrue(False)

if __name__ == "__main__":
    unittest.main()
