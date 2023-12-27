import os
import numpy as np
import pydicom
from sklearn.cluster import KMeans  

path = r'D:\Desktop\cardioid_reorientation\Dataset_params\src_dataset\Server_fromWDY\example\BAI_FUJIN_rest.IMA'

file = pydicom.read_file(path, force=True)
array = file.pixel_array   # .transpose((1,0,2))[:,::-1] -> onn->∪o⊂->∩o⊂
array = np.array(array, dtype=np.float32) # 转类型
print(array.shape)
shape = array.shape
array = 1000 * (array - array.min()) / (array.max() - array.min())

# 重建体数据中的心肌位置一般在固定区域，所以可将其他范围置0
array[:10] = 0  
array[54:] = 0
array[:, :10] = 0
array[:, 55:] = 0
array[:, :, :10] = 0
array[:, :, 54:] = 0

# 获取非零像素的坐标
nonzero_coords = np.column_stack(np.where((array > 0) & (array < 400)))

# 使用K-means++初始化方法，设置2个聚类中心  
kmeans = KMeans(n_clusters=20, init='k-means++', random_state=0).fit(nonzero_coords)  
  
labels = kmeans.labels_  # 每个数据点的聚类标签  
centroids = kmeans.cluster_centers_  # 聚类中心点

# 将聚类结果映射回三维图像
segmentation_result = np.zeros_like(array)
segmentation_result[(array > 0) & (array < 400)] = labels + 1  # 聚类标签从1开始，0表示背景
segmentation_result.astype(np.float32).tofile(path.replace('.IMA', '.dat'))

# # 重塑标签为三维，与原图像形状相同  
# labels_reshaped = labels.reshape(shape)  
# # 将聚类中心作为分割结果（心脏和肝脏的中心位置）  
# segmented_image = np.stack((labels_reshaped == 0, labels_reshaped == 1, labels_reshaped == 2), axis=-1)  
