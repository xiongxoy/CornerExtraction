#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2014-4-29

@author: Zhou Xiong
'''

import unittest
import numpy as np

from research.corner.corner_extraction import CornerExtractor


class Test(unittest.TestCase):

    def setUp(self):
        self.extractor = CornerExtractor([[0, 0]])

    def test_get_index_from_contours_convex(self):
        contour = [[0, 0], [0, 1], [0, 2], [0, 3],
                   [1, 3], [2, 3], [3, 3],
                   [3, 2], [3, 1.5], [3, 1], [3, 0],
                   [2, 0], [1, 0]]
        contour = np.array(contour, np.float32)
        contour = np.vstack(contour).squeeze()
        # assign points to different lines
        idx = self.extractor.get_idx_from_contours_convex(contour, 4)
        self.assertEqual(max(idx) + 1, 4)
        self.assertTrue(idx[0] == idx[1] and idx[1] == idx[2])
        self.assertTrue(idx[3] == idx[4] and idx[4] == idx[5])

    def test_adjust_indexes(self):
        indexes = [2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        ret_indexes = self.extractor.adjust_indexes(indexes)
        expected_indexes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1]
        self.assertTrue(
             np.testing.assert_allclose(ret_indexes, expected_indexes) is None
             )

    def test_interpolate_points_case1(self):
        points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                          dtype=np.float32)
        ret_points = self.extractor.interpolate_points(points, 1)
        expected_points = np.array([[1, 1], [1.5, 1.5], [2, 2],
                                    [2.5, 2.5], [3, 3], [3.5, 3.5],
                                    [4, 4], [4.5, 4.5], [5, 5], [3, 3]],
                                    dtype=np.float32)
        self.assertEqual(len(ret_points), len(expected_points))
        self.assertTrue(
             np.testing.assert_allclose(ret_points, expected_points) is None
             )

    def test_interpolate_points_case2(self):
        points = np.array([[1, 1], [3, 3]], dtype=np.float32)
        ret_points = self.extractor.interpolate_points(points, 2)
        expected_points = np.array(
                              [[1, 1], [1.5, 1.5], [2, 2],
                              [2.5, 2.5], [3, 3], [2.5, 2.5],
                              [2, 2], [1.5, 1.5]], dtype=np.float32
                          )
        self.assertEqual(len(ret_points), len(expected_points))
        self.assertTrue(
             np.testing.assert_allclose(ret_points, expected_points) is None
             )

    def test_indexes_in_window_case1(self):
        indexes = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        ret_indexes = self.extractor.get_indexes_in_window(indexes, 0, 1)
        expected_indexes = [1, 2, 2]
        self.assertItemsEqual(ret_indexes, expected_indexes)

    def test_indexes_in_window_case2(self):
        indexes = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        ret_indexes = self.extractor.get_indexes_in_window(indexes, 0, 2)
        expected_indexes = [1, 2, 2, 2, 2]
        self.assertItemsEqual(ret_indexes,  expected_indexes)

    def test_indexes_in_window_case3(self):
        indexes = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        ret_indexes = self.extractor.get_indexes_in_window(indexes, 2, 2)
        expected_indexes = [2, 2, 2, 0, 0]
        self.assertItemsEqual(ret_indexes,  expected_indexes)

    def test_indexes_in_window_case4(self):
        indexes = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        ret_indexes = self.extractor.get_indexes_in_window(indexes, 18, 1)
        expected_indexes = np.array([2, 2, 1],  dtype=np.float32)
        self.assertItemsEqual(ret_indexes,  expected_indexes)

    def test_filtering_indexes_case1(self):
        indexes = [0, 0, 3, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        ret_indexes = self.extractor.filter_indexes(indexes, 1)
        expected_indexes = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        self.assertTrue(
             np.testing.assert_allclose(ret_indexes,  expected_indexes) is None
             )

    def test_filtering_indexes_case2(self):
        ''' 假设每条边有足够的点 '''
        indexes = [2, 2, 2, 0, 0, 0, 7, 0, 0, 0, 1,
                   1, 8, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        ret_indexes = self.extractor.filter_indexes(indexes,  2)
        expected_indexes = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1,
                            1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
        self.assertTrue(
             np.testing.assert_allclose(ret_indexes,  expected_indexes) is None
             )

if __name__ == "__main__":
    unittest.main()
