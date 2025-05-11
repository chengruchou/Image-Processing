# E24104294
from numpy import array, zeros_like, ones, zeros
from numpy import abs, max, min
from numpy import fft, float32, uint8
from filtering import *
from morphology import *
# --- import the above object/funcs, only. ---#
# !!! Import others package is forbidden.  !!!#

def denoising_func(img:array):
    # Input array size must be the same with output array.
    kernel = ones((3, 3), dtype=float32) / 9.0
    return spatialdomain_filtering(img, kernel).astype(uint8)

def deSaltPepper_func(img:array):
    # Input array size must be the same with output array.
    k = ones((3, 3), dtype=uint8)
    cleaned = opening(img, k)
    cleaned = closing(cleaned, k)
    return cleaned