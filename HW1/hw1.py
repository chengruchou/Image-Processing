import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(channel):
    # 計算直方圖：計算 0~255 的像素出現次數
    hist = np.bincount(channel.flatten(), minlength=256)
    
    # 計算累積分布函數 (CDF)
    cdf = hist.cumsum()
    
    # 將 CDF 正規化到 0-255 範圍
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    
    # 利用正規化的 CDF 當作查找表映射原始像素值
    equalized = cdf_normalized[channel]
    
    return equalized

def main():
    # 讀取輸入圖片，請確保 input.png 與程式在同一資料夾中
    img = cv2.imread('HW1/input.jpg')
    if img is None:
        print("Error: 'input.png' not found.")
        return

    # 轉換 BGR 至 HSV 色彩空間，分離出 H, S, V 三個通道
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # -----------------------
    # 產生 H 與 S 通道的 2D 直方圖
    # -----------------------
    # 計算 2D histogram (bins: Hue 0~180, Saturation 0~256)
    hist2d, xedges, yedges = np.histogram2d(
        h.flatten(), s.flatten(), bins=[180, 256], range=[[0, 180], [0, 256]]
    )
    
    plt.figure(figsize=(6, 6))
    plt.imshow(hist2d.T, origin='lower', aspect='auto', extent=[0, 180, 0, 256])
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.title('2D Histogram for H and S channels')
    plt.colorbar()
    plt.savefig('HW1/2d_histogram.png')
    plt.close()

    # -----------------------
    # 對 V 通道進行直方圖均衡化
    # -----------------------
    v_eq = histogram_equalization(v)
    
    # 合併均衡化後的 V 與原始的 H, S 通道，並轉回 BGR 色彩空間
    hsv_eq = cv2.merge([h, s, v_eq])
    output_img = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    cv2.imwrite('HW1/output.png', output_img)
    
    # -----------------------
    # 生成比較圖：原圖、均衡化後圖與其各自的灰階 PDF
    # -----------------------
    # 將 BGR 轉換為 RGB 方便 matplotlib 顯示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    # 將原圖與結果圖轉為灰階
    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    
    # 計算灰階直方圖 (256 bins)
    hist_orig = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
    hist_eq = cv2.calcHist([gray_equalized], [0], None, [256], [0, 256])
    
    # 轉換為 PDF (機率密度函數)
    pdf_orig = hist_orig / hist_orig.sum()
    pdf_eq = hist_eq / hist_eq.sum()
    
    # 建立 2x2 子圖網格
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # 左上：原圖 (RGB)
    axs[0, 0].imshow(img_rgb)
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Original Image')
    
    # 右上：原圖的灰階 PDF
    axs[0, 1].plot(pdf_orig, color='blue')
    axs[0, 1].set_title('Original Grayscale PDF')
    axs[0, 1].set_xlabel('Gray Level')
    axs[0, 1].set_ylabel('Probability Density')
    
    # 左下：均衡化後圖 (RGB)
    axs[1, 0].imshow(output_rgb)
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Equalized Image')
    
    # 右下：均衡化後圖的灰階 PDF
    axs[1, 1].plot(pdf_eq, color='red')
    axs[1, 1].set_title('Equalized Grayscale PDF')
    axs[1, 1].set_xlabel('Gray Level')
    axs[1, 1].set_ylabel('Probability Density')
    
    plt.tight_layout()
    plt.savefig('HW1/result.png')
    plt.close()

if __name__ == '__main__':
    main()
