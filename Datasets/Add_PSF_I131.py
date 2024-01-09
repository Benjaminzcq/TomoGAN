import os
import numpy as np
import pydicom
import scipy
import matplotlib.pyplot as plt

def Gaussian_filter(zone):
    FWHM = 2
    point = FWHM
    s1 = point/np.sqrt(8*np.log(2))
    radius = np.floor(4*s1)
    data_filter = np.zeros_like(zone)

    x, y = np.meshgrid(np.arange(-radius, radius+1),np.arange(-radius, radius+1))
    h_2d = np.exp(-(x**2 + y**2)/(2 * s1**2))
    h_2d = h_2d/np.sum(h_2d[:])
    print("开始高斯滤波")
    data_filter = scipy.signal.convolve2d(zone[0], h_2d, mode='same')
    data_filter = np.expand_dims(data_filter, axis=0)
    print("高斯滤波结束")
    return data_filter

def equalizehist(psf):
    # 直方图均衡化
    psf_flat = psf.flatten()
    print("psf_flat:  ", min(psf_flat), max(psf_flat))

    hist, bins = np.histogram(psf_flat, bins=256, range=[min(psf_flat), max(psf_flat)])
    # 计算累积分布函数（CDF）
    cdf = hist.cumsum()

    # 根据CDF对直方图进行映射
    equalized_hist = cdf * hist.max() / cdf.max()

    # 将映射后的直方图进行缩放
    equalized_hist_scaled = (equalized_hist - equalized_hist.min()) / (equalized_hist.max() - equalized_hist.min())
    # 直方图均衡化，插值
    equalized_image = np.interp(psf_flat.flatten(), bins[:-1], equalized_hist_scaled)
    print(f"equalized_image:{equalized_image.shape}")
    print(f"equalized_image:{equalized_image.max()}")
    # 将映射后的像素值替换回原始图像的形状
    equalized_image = equalized_image.reshape([1, 256, 256]).astype(np.float32)
    return equalized_image


########################## 导入投影数据 ##############################
src_path = r'D:\Desktop\artifacts_correction\Datasets\src_dataset'
nm_path= os.path.join(src_path, 'Normal\\CHENJIANFENG_YL.dcm')
nm_data = pydicom.read_file(nm_path, force=True)

nm_array = nm_data.pixel_array.astype(np.float32)   # .transpose((1,0,2))[:,::-1] -> onn->∪o⊂->∩o⊂
nm_shape = nm_array.shape  # [2, 1024, 256]

# 获取上半身前面最大值索引
nm_array_half = nm_array[:, :200, :]
max_indices_front = np.where(nm_array_half[0] == np.max(nm_array_half[0]))  # 体前上半身最大值
mcod_fr = list(zip(max_indices_front[0], max_indices_front[1]))[0]  # max coordinates front
print("体前最大值和对应坐标： ", nm_array_half[0][mcod_fr], mcod_fr)
# 获取上半身后面最大值索引
max_indices_back = np.where(nm_array_half[1]  == np.max(nm_array_half[1]))  # 体后上半身最大值
mcod_ba = list(zip(max_indices_back[0], max_indices_back[1]))[0]     # max coordinates back
print("体后最大值和对应坐标： ", nm_array_half[1][mcod_ba], mcod_ba)
# print([id_i_array[i] for i in max_coordinates_front])

########################### 导入PSF数据 ###############################
psf_path = os.path.join(src_path, 'Sinogram_SingleHead600-5cm-368-MainWindow_256f.dat')
psf_data = np.fromfile(psf_path, dtype=np.float32)
psf_array = psf_data.reshape([1, 256, 256])
psf_array[psf_array > 10000] = 10000    # 去除离群不合理值
# psf_array[psf_array < 10] = 0    # 去除离群不合理值

# 纠正PSF方向，并归一化，再乘以目标病灶点值
kernel_equlized = equalizehist(psf_data)
(kernel_equlized*np.max(nm_array_half[0])).tofile(os.path.join(src_path, 'PSF_equal.dat'))
kernel = kernel_equlized.transpose([0,2,1]) ** 2    # 伽马变换(>1,压缩低灰度范围，从而减少PSF四边边缘噪点)
(kernel*np.max(nm_array_half[0])).tofile(os.path.join(src_path, 'PSF_equal_gamma.dat'))
# kernel = np.power(kernel, 2)

# kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 50     # 归一化
# kernel = (kernel - kernel.mean()) / kernel.std()      # z-score
# kernel.tofile(r'D:\Desktop\artifacts_correction\aa.dat')
k_sp = kernel.shape  # k_sp <= [256,256]
kernel = Gaussian_filter(kernel)
print(kernel.shape)

###################### 将kernel适配到体前数据 ##########################
"""
    [mcod_fr[0] + k_sp[1] // 2 <= 328, 不可能超出1024边界，因为已限定上半身]
    比如最大值点在140, 则kernel中心要右移12，裁为[12,256]; 若在120，裁为[0, 248]
    kernel_front_down = 256，因为下界不可能超出1024边界，因为已限定上半身
""" 
# PSF以目标病灶点为中心，扩充到体数据大小
padded_image = np.zeros(nm_shape,dtype=np.float32)

# 计算kernel适配的尺寸
kernel_front_up = 0 if mcod_fr[0] - k_sp[1] // 2 >= 0 else k_sp[1] // 2 - mcod_fr[0]
kernel_front_left = 0 if mcod_fr[1] - k_sp[2] // 2 >= 0 else k_sp[2] // 2 - mcod_fr[1]
kernel_front_right = 256 if mcod_fr[1] + k_sp[2] // 2 <= 256 else 256 - (mcod_fr[1] + k_sp[2] // 2 - 256)
print("\n体前裁剪后的kernel坐标: ",  [kernel_front_up, 256], [kernel_front_left,kernel_front_right])
# 裁剪kernel为适配的尺寸
kernel_front = kernel[:, kernel_front_up:, kernel_front_left: kernel_front_right] * np.max(nm_array_half[0])
                    #   max(mcod_fr[1] - k_sp[2] // 2, 0) : min(mcod_fr[1] + k_sp[2] // 2, 256)] * np.max(nm_array_half[0])
k_sp_new = kernel_front.shape
print("体前裁剪后的kernel尺寸: ",  k_sp_new)
# 计算kernel替换到体数据的起始坐标
pad_front_ud = mcod_fr[0] - k_sp[1] // 2 if mcod_fr[0] - k_sp[1] // 2 >= 0 else 0
pad_front_lr = mcod_fr[1] - k_sp[2] // 2 if mcod_fr[1] - k_sp[2] // 2 >= 0 else 0
print("体前填充kernel的坐标：", [pad_front_ud, pad_front_ud +  k_sp_new[1]], [pad_front_lr, pad_front_lr +  k_sp_new[2]])
# 将适配的kernel代入体数据中
padded_image[:1, pad_front_ud : pad_front_ud +  k_sp_new[1],
             pad_front_lr : pad_front_lr +  k_sp_new[2]] = kernel_front

####################### 将kernel适配到体后数据 ########################
# 计算kernel适配的尺寸
kernel_back_up = 0 if mcod_ba[0] - k_sp[1] // 2 >= 0 else k_sp[1] // 2 - mcod_ba[0]
kernel_back_left = 0 if mcod_ba[1] - k_sp[2] // 2 >= 0 else k_sp[2] // 2 - mcod_ba[1]
kernel_back_right = 256 if mcod_ba[1] + k_sp[2] // 2 <= 256 else 256 - (mcod_ba[1] + k_sp[2] // 2 - 256)
print("\n体后裁剪后的kernel坐标: ",  [kernel_back_up, 256], [kernel_back_left,kernel_back_right])

# 裁剪kernel为适配的尺寸
kernel_back = kernel[:, kernel_back_up:, kernel_back_left: kernel_back_right] * np.max(nm_array_half[0])
k_sp_new1 = kernel_back.shape
print("体后裁剪后的kernel尺寸: ",  k_sp_new1)

# 计算kernel替换到体数据的起始坐标
pad_back_ud = mcod_ba[0] - k_sp[1] // 2 if mcod_ba[0] - k_sp[1] // 2 >= 0 else 0
pad_back_lr = mcod_ba[1] - k_sp[2] // 2 if mcod_ba[1] - k_sp[2] // 2 >= 0 else 0
print("体后填充kernel的坐标：", [pad_back_ud, pad_back_ud + k_sp_new1[1]], [pad_back_lr, pad_back_lr + k_sp_new1[2]])
# 将适配的kernel代入体数据中
padded_image[1:, pad_back_ud : pad_back_ud +  k_sp_new1[1],
             pad_back_lr : pad_back_lr +  k_sp_new1[2]] = kernel_back

######################### 将适配的kernel加入原投影数据 ##########################
result = padded_image + nm_array
# 保存为dcm
nm_data.PixelData = result.astype(np.uint16).tobytes()
nm_data.save_as(nm_path.replace('.dcm', '_conv.dcm').replace('Normal', ''))
result.astype(np.float32).tofile(nm_path.replace('.dcm', '_conv.dat').replace('Normal', ''))


######################### 直方图可视化 ###############################

def hist_vis(array_flat1, array_flat2):
    array_flat1 = array_flat1[0].flatten()
    array_flat2 = array_flat2[0].flatten()
    hist1, bins1 = np.histogram(array_flat1, bins=100, range=[min(array_flat1), max(array_flat1)])
    hist2, bins2 = np.histogram(array_flat2, bins=100, range=[min(array_flat2), max(array_flat2)])

    # 绘制直方图
    plt.figure()    # figsize=(10, 5)
    plt.hist([hist1, hist2], bins=bins2,stacked=True)
    # plt.plot(bins[:-1], hist, color='blue')
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Image')
    plt.show()

hist_vis(nm_array, result)