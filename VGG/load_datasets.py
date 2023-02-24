import sys, os
import cv2 as cv
import numpy as np



class DataLoader():
    def __init__(self, dataset_name, norm_range=(0, 1), img_res=(64, 64)):
        '''
        :param dataset_name: datasets文件夹里边的数据集文件夹名字
        :param norm_range: 输入0-255的图片，归一化为每个像素值在norm_range区间内(含边界)的图片
        :param img_res: 图片形状变为(h,w)。h为单通道图片的高，w为单通道图片的宽。
        '''
        self.dataset_name = dataset_name 
        self.norm_range = norm_range
        self.img_res = img_res

    def img_normalize(self, image, norm_range=(0, 1)):
        '''
        输入0-255的图片，归一化为每个像素值在norm_range区间内(含边界)的图片。
        归一化作用：
        1.便于选择学习率
        2.移除图像的平均亮度值 
        3.让某些激活函数的梯度不致于过小，加快收敛。
        '''
        image = np.array(image).astype('float32')
        image = (norm_range[1] - norm_range[0]) * image / 255. + norm_range[0]  #归一化图像数据要除255
        return image

    def load_datasets(self, dir_path):
        '''
        父目录里边的子目录文件夹为标签，子目录文件夹里边的图片为此标签的数据
        :param dir_path:父目录
        :return:无
        适用于几乎所有图像数据的导入方法
        先给定所有可能的图像格式（也许不全），从文件夹中抽取满足条件的图片类型
        '''
        ImgTypeList = ['jpg', 'JPG', 'bmp', 'png', 'jpeg', 'rgb', 'tif']
        file_names = os.listdir(dir_path)  #指向要读取图片的目录
        x_total_list, y_total_list = [], []
        for i, file_name in enumerate(file_names):
            file_name_path = os.path.join(dir_path, file_name)  
            image_names = os.listdir(file_name_path)  #指向要读取的图像数据的文件名
            x_list, y_list = [], []
            for image_name in image_names:
                if image_name.split('.')[-1] in ImgTypeList:  # 说明是图片，往下操作（文件名与文件类型用.分割）
                    image = cv.imread(os.path.join(file_name_path, image_name), flags=-1)  # flags=-1是以完整的形式读入图片
                    if image.shape[0] != self.img_res[1] or image.shape[1] != self.img_res[0]:
                        image = cv.resize(image,
                                          dsize=(self.img_res[1], self.img_res[0]))  # 如果图片大小与需要裁剪的大小不一样，裁剪图片为指定大小。这里是256*256
                    if len(image.shape) == 2:  # 说明是灰图。如果是灰度图，需要扩展dims。因为灰度图仅仅为单通道
                        image = np.expand_dims(image, axis=-1)
                    x_list.append(image)
                    y_list.append(i)
            x, y = np.array(x_list), np.array(y_list)
            # print(x.shape)
            x_total_list.append(x), y_total_list.append(y)
        x, y = np.concatenate(x_total_list, 0), np.concatenate(y_total_list, 0)  #用concatenate将所有的x和y数据（图像数据与对应的标签数据连接起来）
        # print(x.shape)
        x = self.img_normalize(x, norm_range=(self.norm_range[0], self.norm_range[1]))  # 图像的归一化操作。
        return x, y

    def load_image(self, image_path):  #读取图像的操作
        img = cv.imread(image_path)
        if self.img_res is not None:
            img = cv.resize(img, self.img_res)
        img = img.astype('float32')
        img = self.img_normalize(img, norm_range=(self.norm_range[0], self.norm_range[1]))
        img = img[np.newaxis, :, :, :]
        return img  # (b,h,w,c) 对应一个四维数据。分别为batch，图像的height，图像的width，以及图像通道数
