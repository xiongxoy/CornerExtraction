#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import cv2

from .global_variable import GlobalVariable


# mouse callback function
class RectangleDrawer:
    @staticmethod
    def draw_rectangle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if GlobalVariable.state == 0:
                GlobalVariable.state = 1
                RectangleDrawer.start_rectangle(x, y)
            elif GlobalVariable.state == 1:
                GlobalVariable.state = 2
        elif event == cv2.EVENT_MOUSEMOVE:
            if GlobalVariable.state == 1:
                RectangleDrawer.update_rectangle(x, y)
            if GlobalVariable.state != 2:
                RectangleDrawer.draw_rectangle_on_image()

    @staticmethod
    def start_rectangle(x, y):
        GlobalVariable.start_point = (x, y)
        GlobalVariable.end_point = (x, y)

    @staticmethod
    def update_rectangle(x, y):
        GlobalVariable.end_point = (x, y)

    @staticmethod
    def draw_rectangle_on_image():
        GlobalVariable.display_image = copy.deepcopy(
                                         GlobalVariable.original_image
                                        )
        cv2.rectangle(GlobalVariable.display_image,
                      GlobalVariable.start_point,
                      GlobalVariable.end_point, (0, 0, 255))
        cv2.imshow("image", GlobalVariable.display_image)
