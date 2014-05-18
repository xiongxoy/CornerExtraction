#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2014-4-29

@author: Zhou Xiong
'''

import unittest
import numpy as np

from research.corner.CornerExtration import CornerExtractor

class Test(unittest.TestCase):
    def setUp(self):
        self.extractor = CornerExtractor([[0,0]])
    def tearDown(self):
        pass
    def testCornerExtraction(self):
        pass
    def test_get_index_from_contours_convex(self):
        contour = [[0,0], [0,1], [0,2], [0,3],
                   [1,3], [2,3], [3,3], 
                   [3,2], [3,1.5], [3,1], [3,0], 
                   [2,0], [1,0]]
        contour = np.asarray(contour, 'float32')
        contour = np.vstack(contour).squeeze()
        con_ret, idx = self.extractor.get_idx_from_contours_convex(np.vstack(contour).squeeze(), 4)  # assign points to different lines
#       self.assertEqual(contour, contour)
        print idx
        self.assertEqual(max(idx)+1, 4)
#         self.assertTrue(contour.all(con_ret))
        self.assertTrue( idx[0] == idx[1] and idx[1] == idx[2] )
        self.assertTrue( idx[3] == idx[4] and idx[4] == idx[5] )

    def test_adjust_points(self):
        points = [[0,0], [0,1], [0,2], [0,3],
                  [1,3], [2,3], [3,3], 
                  [3,2], [3,1.5], [3,1], [3,0], 
                  [2,0], [1,0]]
        self.extractor.adjust_points(points, [])

    def test_interpolate_points_case1(self):
        points          = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]], dtype=np.float32)
        ret_points      = self.extractor.interpolate_points(points, 1)
        expected_points = np.array([[1,1], [1.5,1.5], [2,2], [2.5,2.5], [3,3], [3.5,3.5], [4,4], [4.5,4.5], [5,5], [3,3]], dtype=np.float32)
        self.assertEqual(len(ret_points), len(expected_points))
        self.assertTrue(np.testing.assert_allclose(ret_points, expected_points) is None)

    def test_interpolate_points_case2(self):
        points          = np.array([[1,1], [3,3]], dtype=np.float32)
        ret_points      = self.extractor.interpolate_points(points, 2)
        expected_points = np.array([[1,1], [1.5,1.5], [2,2], [2.5,2.5], [3,3], [2.5,2.5], [2,2], [1.5,1.5]], dtype=np.float32)
        self.assertEqual(len(ret_points), len(expected_points))
        self.assertTrue(np.testing.assert_allclose(ret_points, expected_points) is None)
       
    def test_points_in_window(self): 
        points          = np.array([2,2,2,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,1], dtype=np.float32)
        points          = points[:,None] # add 1 dimension to the second dimension  
        ret_points      = self.extractor.get_points_in_window(points,0,1)
        expected_points = np.array([1,2,2], dtype=np.float32)
        expected_points = expected_points[:,None]
#         self.assertTrue(np.testing.a(ret_points, expected_points) is None)
        self.assertItemsEqual(ret_points, expected_points)

    def test_filtering_points_case1(self):
        points     = np.array([0,1,0,1,1,1,2,2,2,2], dtype=np.float32)
        ret_points = np.array([0,0,0,1,1,1,2,2,2,2], dtype=np.float32) 
        self.assertTrue(np.testing.assert_allclose(points, ret_points) is None)

    def test_filtering_points_case2(self):
        ''' 假设每条边有足够的点 '''
        points     = np.array([2,2,2,0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,1], dtype=np.float32)
        ret_poinst = self.extractor.filter_points(points, 2)
        expected_points = np.array([2,2,2,0,0,0,2,1,0,0,1,1,1,1,1,2,2,2,1], dtype=np.float32) 

        self.assertTrue(np.testing.assert_allclose(points, expected_points) is None)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCornerExtraction'
    unittest.main()