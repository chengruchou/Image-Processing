# E24104294
from numpy import array, zeros_like, ones, zeros
from numpy import abs, max, min
from numpy import float32, uint8
#--- import the above object/funcs, only. ---#
#!!! Import others package is forbidden.  !!!#

def my_padding(img:array, size:int, mode:int, value=None):
    # Input array size must be the same with output array.
    # mode == 0 => outside or value
    # mode == 1 => 0
    # mode == 2 => 1
    # mode == 3 => mirror 
    # mode == 4 => repeat
    if isinstance(size, int):
        ph = pw = size
    else:
        ph, pw = size

    H, W = img.shape
    outH, outW = H + 2 * ph, W + 2 * pw

    # 先依照 mode 建立整張輸出矩陣
    if mode == 0:
        pad_val = 0 if value is None else value
        padded = ones((outH, outW), dtype=img.dtype) * pad_val
    elif mode == 1:
        padded = zeros((outH, outW), dtype=img.dtype)        # 全 0
    elif mode == 2:
        padded = ones((outH, outW), dtype=img.dtype)         # 全 1
    else:
        padded = zeros((outH, outW), dtype=img.dtype)        # mirror / repeat 先清空

    # 先把原圖貼進中央
    padded[ph:ph + H, pw:pw + W] = img

    # mirror padding
    if mode == 3:
        # 上、下
        for i in range(ph):
            padded[i, pw:pw + W] = img[ph - i - 1, :] if i < H else img[0, :]
            padded[outH - i - 1, pw:pw + W] = img[H - (ph - i), :] if i < H else img[-1, :]

        # 左、右（含角落）
        for j in range(pw):
            padded[:, j] = padded[:, 2 * pw - j]             # 左
            padded[:, outW - j - 1] = padded[:, outW - 2 * pw + j - 1]  # 右

    # repeat padding（edge‑replicate）
    elif mode == 4:
        # 上下
        padded[:ph, pw:pw + W] = img[0:1, :].repeat(ph, axis=0)
        padded[ph + H:, pw:pw + W] = img[-1:, :].repeat(ph, axis=0)
        # 左右 + 四個角
        padded[:, :pw] = padded[:, pw:pw + 1].repeat(pw, axis=1)
        padded[:, pw + W:] = padded[:, pw + W - 1:pw + W].repeat(pw, axis=1)

    return padded
    return result

#### Problem 1
def dilation(img:array, kernel:array):
    # Input array size must be the same with output array.
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = my_padding(img, (ph, pw), mode=1)          # zero‑padding
    H, W = img.shape
    out = zeros_like(img)

    for y in range(H):
        for x in range(W):
            window = padded[y:y + kh, x:x + kw] * kernel
            out[y, x] = max(window)                     # numpy.max 已在 import
    return out


def erosion(img:array, kernel:array):
    # Input array size must be the same with output array.
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = my_padding(img, (ph, pw), mode=1)
    H, W = img.shape
    out = zeros_like(img)

    for y in range(H):
        for x in range(W):
            window = padded[y:y + kh, x:x + kw]
            # 只考慮 kernel==1 的位置
            vals = [window[i, j] for i in range(kh) for j in range(kw) if kernel[i, j]]
            out[y, x] = min(vals) if vals else 0
    return out

#### Problem 2
def opening(img:array, kernel:array):
    # Input array size must be the same with output array.
    eroded  = erosion(img, kernel)
    opened  = dilation(eroded, kernel)
    return opened

def closing(img:array, kernel:array):
    # Input array size must be the same with output array.
    dilated = dilation(img, kernel)
    closed  = erosion(dilated, kernel)
    return closed

if __name__ == "__main__":

    kernel = array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    image = array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])


    dilated = dilation(image, kernel)
    eroded = erosion(image, kernel)

    print("Original:\n", image)
    print("Dilated:\n", dilated)
    print("Eroded:\n", eroded)