import streamlit as st
import cv2
import numpy as np


class Filtering:
    def __init__(self, in_buffer, in_channels):
        self.origBuffer   = in_buffer
        if self.origBuffer is not None:
            self.origImage = self.origBuffer.read()
        else:
            self.origImage = None
        self.origChannels = in_channels
        self.outImage     = self.get_cv2_image()
        self.outChannels  = ""
        self.kernel_size  = None

    def get_buffer(self):
        return self.origBuffer

    def get_image(self):
        return self.origImage

    def get_cv2_image(self):
        raw_bytes = np.asarray(bytearray(self.origImage), dtype = np.uint8)
        return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    def get_filtered(self):
        return self.outImage, self.outChannels

    def set_orig_img(self):
        self.origImage = self.origBuffer.read()

    def noneFilter(self):
        self.outImage    = self.get_cv2_image()
        self.outChannels = self.origChannels

    def blackWhite(self):
        self.outChannels = "GRAY"
        self.outImage = cv2.cvtColor(self.get_cv2_image(), cv2.COLOR_BGR2GRAY)

    def selectChannel(self, in_channel):
        self.outChannels = "BGR"
        self.outImage = self.get_cv2_image()
        if in_channel == "Blue":
            self.outImage[:, :, 1:] = 0
        elif in_channel == "Green":
            self.outImage[:, :, (0, 2)] = 0
        else:
            self.outImage[:, :, :2] = 0

    def equalizeImg(self, in_color_space):
        self.outChannels = "BGR"
        self.outImage = self.get_cv2_image()
        if in_color_space == "BGR":
            for i in range(3):
                self.outImage[:, :, i] = cv2.equalizeHist(self.outImage[:, :, i])
        else:
            self.outImage = cv2.cvtColor(self.outImage, cv2.COLOR_BGR2HSV)
            self.outImage[:, :, 2] = cv2.equalizeHist(self.outImage[:, :, 2])
            self.outImage = cv2.cvtColor(self.outImage, cv2.COLOR_HSV2BGR)

    def linearBlur(self, in_kernel):
        self.outChannels = "BGR"
        if in_kernel is not None:
            self.outImage = cv2.filter2D(self.get_cv2_image(),
                                         ddepth = -1,
                                         kernel = in_kernel.get_kernel())

    def gaussianBlur(self, in_kernel):
        self.outChannels = "BGR"
        if in_kernel is not None:
            self.outImage = cv2.GaussianBlur(self.get_cv2_image(),
                                             in_kernel.get_kernel_size(),
                                             in_kernel.get_kernel_sigma())

    def medianBlur(self, in_median):
        self.outChannels = "BGR"
        self.outImage    = cv2.medianBlur(self.get_cv2_image(), in_median)

    def bilateralBlur(self, in_diameter, in_sigma_color, in_sigma_space):
        self.outChannels = "BGR"
        self.outImage    = cv2.bilateralFilter(self.get_cv2_image(),
                                               in_diameter,
                                               in_sigma_color,
                                               in_sigma_space)

    def sharpening(self, in_kernel):
        self.outChannels = "BGR"
        if in_kernel is not None:
            self.outImage = cv2.filter2D(self.get_cv2_image(),
                                         ddepth = -1,
                                         kernel = in_kernel.get_kernel())
            self.outImage = cv2.cvtColor(self.outImage, cv2.COLOR_RGB2BGR)

    def sobel(self, in_kernel):
        self.outChannels = "GRAY"
        if in_kernel is not None:
            self.outImage = cv2.cvtColor(self.get_cv2_image(),
                                         cv2.COLOR_BGR2GRAY)
            self.outImage = np.clip(cv2.filter2D(self.outImage,
                                                 ddepth     = cv2.CV_64F,
                                                 kernel     = in_kernel.get_kernel()),
                                    0,
                                    255).astype("uint8")

    def canny(self, in_lt, in_ut):
        self.outChannels = "GRAY"
        self.outImage = cv2.cvtColor(self.get_cv2_image(),
                                     cv2.COLOR_BGR2GRAY)
        self.outImage = cv2.Canny(self.outImage,
                                  threshold1 = in_lt,
                                  threshold2 = in_ut)

    def applyFilter(self, in_filter_type, in_kernel = None):
        if in_filter_type   == "None":
            self.noneFilter()
        elif in_filter_type == "Black&White":
            self.blackWhite()
        elif in_filter_type == "Red Channel" or in_filter_type == "Blue Channel" or in_filter_type == "Green Channel":
            self.selectChannel(in_filter_type[:in_filter_type.find(" ")])
        elif in_filter_type == "Equalization":
            self.equalizeImg(st.radio("Select Color Space",
                                      ["BGR", "HSV"],
                                      index = 0))
        elif in_filter_type == "Linear Blur":
            self.linearBlur(in_kernel)
        elif in_filter_type == "Gaussian Blur":
            self.gaussianBlur(in_kernel)
        elif in_filter_type == "Median Blur":
            self.medianBlur(st.slider(label     = "Median Blur center select",
                                      min_value = 1,
                                      max_value = 31,
                                      step      = 2))
        elif in_filter_type == "Bilateral Blur":
            self.bilateralBlur(st.slider(label     = "Bilateral Blur diameter select",
                                         min_value = 1,
                                         max_value = 30,
                                         step      = 1),
                               st.slider(label     = "Bilateral Blur sigma color select",
                                         min_value = 1,
                                         max_value = 255,
                                         step      = 1),
                               st.slider(label     = "Bilateral Blur sigma space select",
                                         min_value = 1,
                                         max_value = 255,
                                         step      = 1))
        elif in_filter_type == "Sharpened":
            self.sharpening(in_kernel)
        elif in_filter_type == "Sobel":
            self.sobel(in_kernel)
        elif in_filter_type == "Canny":
            lt = st.slider(label     = "Canny Lower Threshold",
                           min_value = 1,
                           max_value = 253,
                           step      = 1)
            ut = st.slider(label     = "Canny Upper Threshold",
                           min_value = lt + 1,
                           max_value = 255,
                           value     = int((lt + 1 + 255) / 2),
                           step      = 1)
            self.canny(lt, ut)
