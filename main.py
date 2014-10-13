#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2014-7-2

@author:  Zhou Xiong
'''
import cv2
import logging
import copy

from research.corner.corner_extraction import get_sub_image, CornerExtractor
from research.corner.global_variable import GlobalVariable
from research.util import info


def perspectvie_trans_demo():
    from research.corner.image_transformer import ImageTransformer
    from research.corner.rectangle_drawer import RectangleDrawer

    # Create a black image, a window and bind *draw_rectangle* to it
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', RectangleDrawer.draw_rectangle)

    # Read image
    GlobalVariable.original_image = cv2.imread('img/corridor.jpg')
    if GlobalVariable.original_image is None:
        logging.error('Image Not Found')
        assert 0
    GlobalVariable.original_image = cv2.resize(GlobalVariable.original_image,
                                               (0, 0), fx=0.3, fy=0.3)
    # image for display
    GlobalVariable.display_image = copy.deepcopy(GlobalVariable.original_image)
    # Show image
    cv2.imshow("image", GlobalVariable.display_image)
    # Wait for user input,
    # the target rectangle should be drawn before hitting caret
    key = cv2.waitKey(0)
    # if caret is hit
    if key == ord('\r'):
        # extract target region
        sub_image = get_sub_image(GlobalVariable.original_image,
                                  GlobalVariable.start_point,
                                  GlobalVariable.end_point)
        # transform image
        transformer = ImageTransformer(sub_image)
        transformed_image = transformer.transform()
        # show result and wait
        cv2.imshow("transformed", transformed_image)
        cv2.waitKey(0)

    # clean up windows
    cv2.destroyAllWindows()


def ransac_corner_extraction_demo(path, n, convex):
    from research.util import Plotter
    img = cv2.imread(path)
    GlobalVariable.original_image = img
    extractor = CornerExtractor(img)
    vertices = extractor.extract(n, convex)
    info(vertices)
    Plotter.plot_points(GlobalVariable.original_image,
                        vertices, 'final points')

if __name__ == '__main__':
    info('Hello')
    info("Good Start")
    perspectvie_trans_demo()
#     ransac_corner_extraction_demo('img/flag.jpg', 7, False)
#     ransac_corner_extraction_demo('img/rect.jpg', 4, False)
#     ransac_corner_extraction_demo('img/array.jpg', 4, False)
    info('Bye')
