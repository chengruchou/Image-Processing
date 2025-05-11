# E24104294
from numpy import array, zeros_like, ones, zeros
from numpy import abs, max, min
from numpy import float32, uint8, fft
#--- import the above object/funcs, only. ---#
#!!! Import others package is forbidden.  !!!#

#### Problem 1
def my_padding(input:array, size:int, mode:int, value=None):
    # Input array size must be the same with output array.
    # mode == 0 => outside or value
    # mode == 1 => 0
    # mode == 2 => 1
    # mode == 3 => mirror 
    # mode == 4 => repeat
    ph, pw = (size, size) if isinstance(size, int) else size
    H, W   = input.shape
    outH, outW = H + 2*ph, W + 2*pw

    if   mode == 0:
        pad_val = 0 if value is None else value
        padded  = ones((outH, outW), dtype=input.dtype) * pad_val
    elif mode == 1:
        padded  = zeros((outH, outW), dtype=input.dtype)
    elif mode == 2:
        padded  = ones((outH, outW),  dtype=input.dtype)
    else:                               # mirror / repeat ─ 先留空
        padded  = zeros((outH, outW),  dtype=input.dtype)

    padded[ph:ph+H, pw:pw+W] = input           # 先把原圖貼上去

    if mode == 3:                              # mirror
        for i in range(ph):
            padded[i,           pw:pw+W] = input[ph-i-1 if i < H else 0,  :]
            padded[outH-i-1,    pw:pw+W] = input[H-(ph-i) if i < H else H-1, :]
        for j in range(pw):
            padded[:, j]            = padded[:, 2*pw-j]
            padded[:, outW-j-1]     = padded[:, outW-2*pw+j-1]
    elif mode == 4:                            # repeat edge
        padded[:ph,        pw:pw+W] = input[0:1, :].repeat(ph, axis=0)
        padded[ph+H:,      pw:pw+W] = input[-1:, :].repeat(ph, axis=0)
        padded[:, :pw]              = padded[:, pw:pw+1].repeat(pw, axis=1)
        padded[:, pw+W:]            = padded[:, pw+W-1:pw+W].repeat(pw, axis=1)

    return padded


def my_2d_conv(image:array, kernal:array):
    kh, kw = kernal.shape
    ph, pw = kh // 2, kw // 2
    padded = my_padding(image, (ph, pw), mode=0)
    H, W   = image.shape
    out    = zeros_like(image, dtype=float32)

    for y in range(H):
        for x in range(W):
            window      = padded[y:y+kh, x:x+kw]
            out[y, x]   = (window * kernal).sum()
    return out

#### Problem 2
def spatialdomain_filtering(image:array, kernel:array):
    # Input array size must be the same with output array.
    return my_2d_conv(image.astype(float32), kernel.astype(float32))

def freqencydomain_filtering(image:array, kernel:array):
    # Input array size must be the same with output array.
    H, W  = image.shape
    kh, kw = kernel.shape

    pad_ker           = zeros_like(image, dtype=float32)
    pad_ker[:kh, :kw] = kernel.astype(float32)
    pad_ker           = fft.ifftshift(pad_ker)          # 使 kernel 中心對齊 (0,0)

    F_img    = fft.fft2(image.astype(float32))
    F_kernel = fft.fft2(pad_ker)
    F_out    = F_img * F_kernel

    result = fft.ifft2(F_out).real
    return result.astype(float32), F_img, F_kernel

def spatial_hybrid_imaging(for_lowpass:array, for_highpass:array, kernel=None):
    # Input array size must be the same with output array.
    ############################################
    # kernel should be imported from outside,
    # or defined by yourself if we don't specify
    ############################################
    if kernel is None:
        kernel = ones((9, 9), dtype=float32) / 81.0      # 9×9 mean blur
    low  = spatialdomain_filtering(for_lowpass, kernel)
    blur = spatialdomain_filtering(for_highpass, kernel)
    high = for_highpass.astype(float32) - blur
    hybrid = low + high
    return hybrid.astype(float32), low.astype(float32), high.astype(float32)

def freq_hybrid_imaging(for_lowpass:array, for_highpass:array, kernel=None):
    # Input array size must be the same with output array.
    ############################################
    # kernel should be imported from outside,
    # or defined by yourself if we don't specify
    ############################################
    if kernel is None:
        kernel = ones((9, 9), dtype=float32) / 81.0
    low, _, _  = freqencydomain_filtering(for_lowpass,  kernel)
    blur, _, _ = freqencydomain_filtering(for_highpass, kernel)
    high   = for_highpass.astype(float32) - blur
    hybrid = low + high
    return hybrid.astype(float32), low.astype(float32), high.astype(float32)

