import numpy as np
import streamlit as st


class Kernels:
    def __init__(self):
        self.kernel_width  = 0
        self.kernel_height = 0
        self.kernel_size   = (self.kernel_width, self.kernel_height)
        self.kernel_type   = None
        self.kernel        = None
        self.sigma         = 0
        self.lower_thresh  = 0
        self.upper_thresh  = 0

    def set_blurBox(self, in_size):
        self.kernel_width  = in_size
        self.kernel_height = in_size
        self.kernel_size   = (self.kernel_width, self.kernel_height)
        self.kernel_type   = "BlurBox"
        self.kernel        = np.ones(self.kernel_size, dtype = np.float32) / (self.kernel_width * self.kernel_height)

    def set_Gaussian(self, in_size, in_sigma):
        self.kernel_width  = in_size
        self.kernel_height = in_size
        self.kernel_size   = (self.kernel_width, self.kernel_height)
        self.kernel_type   = "Gaussian"
        self.kernel        = None
        self.sigma         = in_sigma

    def set_Sharpen(self, in_intensity):
        self.kernel_width  = 3
        self.kernel_height = 3
        self.kernel_size   = (self.kernel_width, self.kernel_height)
        self.kernel_type   = "Sharpen"
        k = -(in_intensity - 1) / 4.0
        self.kernel        = np.array([[0, k, 0],
                                       [k, in_intensity, k],
                                       [0, k, 0]], dtype = np.float32)

    def set_Sobel(self, in_axis):
        self.kernel_width  = 0
        self.kernel_height = 0
        self.kernel_size   = (self.kernel_width, self.kernel_height)
        self.kernel_type   = "Sobel " + in_axis
        self.kernel        = np.array([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]])
        if in_axis == "y":
            self.kernel = self.kernel.T

    def set_kernel(self, in_type):
        if in_type == "Linear Blur":
            self.set_blurBox(st.slider(label     = "Linear Blur kernel size select",
                                       min_value = 3,
                                       max_value = 30))
        elif in_type == "Gaussian Blur":
            self.set_Gaussian(st.slider(label     = "Gaussian Blur kernel size select",
                                        min_value = 3,
                                        max_value = 33,
                                        step      = 2),
                              st.slider(label     = "Gaussian Blur sigma select",
                                        min_value = 0.1,
                                        max_value = 10.0,
                                        step      = 0.1))
        elif in_type == "Sharpened":
            self.set_Sharpen(st.slider(label     = "Sharpening intensity",
                                       min_value = 1,
                                       max_value = 30,
                                       step      = 1))
        elif in_type == "Sobel":
            self.set_Sobel(st.radio("Select Sobel Axis",
                                    ["x", "y"],
                                    index = 0))

    def get_kernel(self):
        return self.kernel

    def get_kernel_type(self):
        return self.kernel_type

    def get_kernel_size(self):
        return self.kernel_size

    def get_kernel_sigma(self):
        return self.sigma
