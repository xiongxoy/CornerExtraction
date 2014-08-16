'''
Created on 2014-8-16

@author: zx
'''
import unittest
import numpy as np
from research.util import interpolate_points, get_elements_in_window


class Test(unittest.TestCase):

    def test_interpolate_points_case1(self):
        points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                          dtype=np.float32)
        ret_points = interpolate_points(points, 1)
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
        ret_points = interpolate_points(points, 2)
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
        ret_indexes = get_elements_in_window(indexes, 0, 1)
        expected_indexes = [1, 2, 2]
        self.assertItemsEqual(ret_indexes, expected_indexes)

    def test_indexes_in_window_case2(self):
        indexes = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        ret_indexes = get_elements_in_window(indexes, 0, 2)
        expected_indexes = [1, 2, 2, 2, 2]
        self.assertItemsEqual(ret_indexes,  expected_indexes)

    def test_indexes_in_window_case3(self):
        indexes = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        ret_indexes = get_elements_in_window(indexes, 2, 2)
        expected_indexes = [2, 2, 2, 0, 0]
        self.assertItemsEqual(ret_indexes,  expected_indexes)

    def test_indexes_in_window_case4(self):
        indexes = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        ret_indexes = get_elements_in_window(indexes, 18, 1)
        expected_indexes = np.array([2, 2, 1],  dtype=np.float32)
        self.assertItemsEqual(ret_indexes,  expected_indexes)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()