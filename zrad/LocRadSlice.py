# -*- coding: utf-8 -*-
""" ************* Slice.py """""""""""""""""" 
    This class provides all slice related data and functionality

    Objects of this class contain:
        - Slice location            - Relative position of the image plane expressed in mm
        - Pixel data                - The image matrix
        - ROI Pixel data            - The image matrix within the drawn contours
        - Contour coordinates [mm]  - Number of columns in the image
        - Contour coordinates [px]  - The labels of the contoured structures (e.g. "GTV-1")

    (c) Ambrusch Andreas, ETH
"""

import numpy as np


class Slice:

    def __init__(self, slice_location, px_data):
        self.slice_location = slice_location
        self.pixel_data = px_data
        self.pixel_data_roi = []

    def __str__(self):
        s = "SliceLocation: " + str(self.slice_location) + "\nPixelDataLength: " + str(len(self.pixel_data))
        return s

