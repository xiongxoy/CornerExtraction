'''
Created on 2014-8-16

@author: zx
'''
import unittest
import numpy as np
from research.util import interpolate_points, get_elements_in_window,\
    convert_chiancode, Plotter
import cv2


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

    def test_plot_chain_code(self):
        code = '0007776555444333100222'
        points = convert_chiancode(code, 20)
        img = np.zeros((500, 500), dtype=np.uint8)
        Plotter.plot_points(img, points)

    @unittest.skip('')
    def test_cv2_gray_to_rgb(self):
        img_gray = np.zeros((100, 100), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        color = (0, 0, 255)
        cv2.circle(img_bgr, (50, 50), 10, color, 2)
        cv2.imshow('Color Image', img_bgr)
        cv2.waitKey(0)

    @unittest.skip('')
    def test_cv2_rgb_to_gray(self):
        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray[50, 50:] = 255
        cv2.circle(img_gray, (50, 50), 10, 255, 2)
        cv2.imshow('Gray Image', img_gray)
        cv2.waitKey(0)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
