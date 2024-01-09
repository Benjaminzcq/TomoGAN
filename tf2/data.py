import numpy as np 
import h5py, threading
import queue as Queue
import h5py, glob, os
from util import scale2uint8
import pydicom
import random

class bkgdGen(threading.Thread):
    """
    实现一个线程，用于在后台生成数据，并将生成的数据存储在队列中。
    """
    def __init__(self, data_generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = data_generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            # block if necessary until a free slot is available
            self.queue.put(item, block=True, timeout=None)
        self.queue.put(None)

    def next(self):
        # block if necessary until an item is available
        next_item = self.queue.get(block=True, timeout=None)
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

def gen_train_batch_bg(dsfn, mb_size, in_depth, img_size):
    with h5py.File(dsfn, 'r') as h5fd:
        X = h5fd['train_ns'][:].astype(np.float32)
        Y = h5fd['train_gt'][:].astype(np.float32)

    while True:
        idx = np.random.randint(0, X.shape[0]-in_depth, mb_size)
        if img_size == X.shape[1]:
            rst, cst = np.zeros(mb_size, dtype=np.int), np.zeros(mb_size, dtype=np.int)
        else:
            rst = np.random.randint(0, X.shape[1]-img_size, mb_size)
            cst = np.random.randint(0, X.shape[2]-img_size, mb_size)

        batch_X = np.array([np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)) for s_idx in idx])
        batch_X = [batch_X[_i, _r:_r+img_size, _c:_c+img_size, :] for _i, _r, _c in zip(range(mb_size), rst, cst)]

        batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 3)
        batch_Y = [batch_Y[_i, _r:_r+img_size, _c:_c+img_size, :] for _i, _r, _c in zip(range(mb_size), rst, cst)]

        yield np.array(batch_X), np.array(batch_Y)  # 可生成一组np数据的迭代器

# def gen_train_batch_bg(dsfn, mb_size, in_depth):
#     """
#         生成训练数据集的批次，每次迭代返回mb_size个数据
#     """
#     train_ns_path = os.path.join(dsfn, 'imagesTr')
#     train_gt_path = os.path.join(dsfn, 'labelsTr')

#     # 读取训练数据集中所有dcm文件
#     ns_slices = [pydicom.read_file(s, force=True) for s in glob.glob(os.path.join(train_ns_path, '*.dcm'))]
#     gt_slices = [pydicom.read_file(s, force=True) for s in glob.glob(os.path.join(train_gt_path, '*.dcm'))]

#     # 划分训练数据集并转为array
#     ns_array = np.stack([s.pixel_array.astype(np.float32) for s in ns_slices])  # (N, 2, 1024, 256)
#     gt_array = np.stack([s.pixel_array.astype(np.float32) for s in gt_slices])  # (N, 2, 1024, 256)

#     # 生成训练数据集的批次
#     while True:
#         #### choice从给定的数组中随机生成一组索引，randint从给定的上下限中生成一个包含随机索引的整数数组。
#         #### 范围不大时，两个函数均可使用；两个函数生成的索引中均有重复数据
#         # idx = np.random.choice(len(ns_array)-in_depth+1, mb_size) # 索引有重复
#         idx = np.random.randint(0, len(ns_array)-in_depth, mb_size) # 索引有重复
#         # idx = random.sample(range(len(ns_array)-in_depth+1), mb_size)   # 索引不重复
#         batch_X = np.array([ns_array[s_idx : (s_idx + in_depth)] for s_idx in idx])
#         batch_X = np.transpose(batch_X, [1,2,3,0]) # batch_X->(2,1024,256,mb_size), mb_size是batch数
#         batch_Y = np.array([gt_array[s_idx : (s_idx + in_depth)] for s_idx in idx])
#         batch_Y = np.transpose(batch_Y, [1,2,3,0]) # batch_Y->(2,1024,256,mb_size)
#         yield batch_X, batch_Y  # 可生成一组np数据的迭代器


def get1batch4test(dsfn, in_depth):
    with h5py.File(dsfn, 'r') as h5fd:
        X = h5fd['test_ns'][:].astype(np.float32)
        Y = h5fd['test_gt'][:].astype(np.float32)

    idx = (X.shape[0]-in_depth, ) # always use slice in_depth//2 for validation
    batch_X = np.array([np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)) for s_idx in idx])
    batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 3) 

    return batch_X.astype(np.float32) , batch_Y.astype(np.float32)

# def get1batch4test(dsfn, in_depth):
#     """
#         生成测试数据集的批次，每次迭代返回mb_size个数据
#     """
#     test_ns_path = os.path.join(dsfn, 'imagesTs')
#     test_gt_path = os.path.join(dsfn, 'labelsTs')

#     # 读取测试数据集中所有dcm文件
#     ns_slices = [pydicom.read_file(s, force=True) for s in glob.glob(os.path.join(test_ns_path, '*.dcm'))]
#     gt_slices = [pydicom.read_file(s, force=True) for s in glob.glob(os.path.join(test_gt_path, '*.dcm'))]

#     # 划分测试数据集并转为array
#     ns_array = np.stack([s.pixel_array.astype(np.float32) for s in ns_slices])  # (N, 2, 1024, 256)
#     gt_array = np.stack([s.pixel_array.astype(np.float32) for s in gt_slices])  # (N, 2, 1024, 256)

#     idx = (len(ns_array)-in_depth, ) # always use slice in_depth//2 for validation
#     batch_X = np.array([ns_array[s_idx : (s_idx + in_depth)] for s_idx in idx])
#     batch_X = np.transpose(batch_X, [1,2,3,0]) # batch_X->(2,1024,256,mb_size), mb_size是batch数
#     batch_Y = np.array([gt_array[s_idx : (s_idx + in_depth)] for s_idx in idx])
#     batch_Y = np.transpose(batch_Y, [1,2,3,0]) # batch_Y->(2,1024,256,mb_size)
#     return batch_X , batch_Y
