import sys, os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets

def cv_imread(img_path, BGRAtoBGR=True, flags=-1):
    '''
    输入图片路径，输出(h,w,c)的完整图片。BGRAtoBGR=True时BGRA图片会自动转为BGR（只能读图片，可支持中文路径）
    flags=-1完整读图，0灰图，1为BGR格式。
    这里的函数使得图片的格式可以采用多样化读取，BGR格式同样可以读取。
    '''
    img = cv.imread(img_path, flags=flags)
    if img is None:
        img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags)  # 可支持中文路径
    if img is not None and BGRAtoBGR and img.shape[-1] >= 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)  # 32位深BGRA图片转为24位BGR图
    return img

class DataLoader():
    def __init__(self, dir_path, dataset_name, norm_range=(0, 1), img_res=(64, 64)):
        '''
        :param dataset_name: datasets文件夹里边的数据集文件夹名字。
        :param norm_range: 输入0-255的图片，归一化为每个像素值在norm_range区间内(含边界)的图片。
        :param img_res: 图片形状变为(h,w)的格式（单通道内的宽与高） 若为None则不操作
        '''
        self.dataset_name = dataset_name
        self.norm_range = norm_range
        self.img_res = img_res
        self.image_path = []  # 先把图片的路径全部保存起来

        ''' 
        :此处为一个适用于多种图片的图片导入格式。先列举出所有可能的图片格式，从可能的图片格式中寻找文件夹中包含的。
        再进行图片数据集的导入。
        '''
        ImgTypeList = ['jpg', 'JPG', 'bmp', 'png', 'jpeg', 'rgb', 'tif']
        try:
            file_names = os.listdir(dir_path)
            for dir_path, dir_names, image_names in os.walk(dir_path):
                print(dir_path, dir_names, image_names)
                for image_name in image_names:
                    if image_name.split('.')[-1] in ImgTypeList:  #判断是否为图像的操作（.用于分割图像名与格式名）。
                        image_path = os.path.join(dir_path, image_name)  #连接此前的路径与图像名称路径，形成完整路径。
                        self.image_path.append(image_path)  #满足条件，说明是图片，就导入路径
            self.image_path = np.array(self.image_path)  #将path变为array格式，便于后续的操作
            print(self.image_path)
        except:
            pass
    def img_normalize(self, image, norm_range=(0, 1)):
        '''
        输入0-255的图片，归一化为每个像素值在norm_range区间内(含边界)的图片。
        图像归一化的原因：
        1.找出图像中的那些不变量，使得图像可以抵抗几何变换的攻击
        2.归一化是为了加快训练网络的收敛性
        '''
        image = np.array(image).astype('float32')
        image = (norm_range[1] - norm_range[0]) * image / 255. + norm_range[0]
        return image

    def load_datasets(self, ):
        '''
        父目录里边的子目录文件夹为标签，子目录文件夹里边的图片为此标签的数据。
        :param dir_path:表示父目录
        :return:该函数为载入数据集函数，故没有返回值
        '''
        x_list = []
        for image_path in self.image_path:
            image = cv_imread(image_path, flags=-1)  # flags=-1是以完整的形式读入图片
            if self.img_res is not None and (
                    image.shape[0] != self.img_res[1] or image.shape[1] != self.img_res[0]):
                image = cv.resize(image,
                                  dsize=(self.img_res[1], self.img_res[0]))  # 如果图片大小与需要裁剪的大小不一样，则裁剪图片为指定大小。
                                                                             # 裁剪图片的大小后续设定（这里默认为128*128）
            if len(image.shape) == 2:  # 说明是灰图，跳过
                # print('k'*200)
                # image = np.expand_dims(image, axis=-1)
                # if image.shape[-1]==3:
                #     image=np.repeat(image,repeats=3,axis=-1)
                continue
            #print(image.shape)
            x_list.append(image)  #在x_list中添加image。
        #self.x = np.array(x_list)
        self.x = np.stack(x_list, axis=0)  #图像数据进行0维堆叠，就是把它们按顺序排放了起来进行存储。
        if self.norm_range is not None:   #当归一化参数并非没有时
            self.x = self.img_normalize(self.x, norm_range=self.norm_range)  #对所有的图像依次进行归一化操作。
        return self.x


    def load_batch(self, batch_size=128):  # 加载一次batch_size=128
        '''
        # 加载一次batch_size=128耗时0.00498秒。这一时间并不长。
        :return: x_batch, y_batch
        '''
        x_list = []
         #从已经载入的所有图像数据当中，任意抽取batch_size数量的图像进行一个batch的训练
        train_ids = np.random.randint(0, len(self.image_path), size=batch_size)
        for image_path in self.image_path[train_ids]:
            image = cv_imread(image_path, flags=-1)  # flags=-1是以完整的形式读入图片
            if self.img_res is not None and (
                    image.shape[0] != self.img_res[1] or image.shape[1] != self.img_res[0]):
                # 如果图片大小与需要裁剪的大小不一样，裁剪图片为指定大小。
                image = cv.resize(image,
                                  dsize=(self.img_res[1], self.img_res[0]))  # 如果图片大小与需要裁剪的大小不一样，裁剪图片为指定大小
            if len(image.shape) == 2:  # shape=2说明是灰度图不为彩图，这里的模型是针对彩图的，因此跳过
                # print('k'*200)
                # image = np.expand_dims(image, axis=-1)
                # if image.shape[-1]==3:
                #     image=np.repeat(image,repeats=3,axis=-1)
                continue
            x_list.append(image)   #在x_list中添加image。
        #x_batch = np.array(x_list)
        x_batch=np.stack(x_list,axis=0)  #一个batch中的图像数据进行0维堆叠，就是把它们按顺序排放了起来进行存储。
        if self.norm_range is not None:
            x_batch = self.img_normalize(x_batch, norm_range=self.norm_range)  #对所有的图像依次进行归一化操作。
        return x_batch


def main():  #可以用于导入数据类DataLoader的测试
    height = 64
    width = 64
    # image_path = r'C:\Users\Administrator\Desktop\seeprettyface_wanghong'
    # from_dir_resize_image(image_path, height, width, save=True, to_save=r'C:\Users\Administrator\Desktop\star_images')
    dataset_name = 'female'  # 数据集名字
    dir_path = r'.\datasets\%s' % dataset_name  # 训练集图片
    #dataloader进行数据的有效裁剪（设定img_res，这里为h=w=128）
    dataloader = DataLoader(dir_path, dataset_name=dataset_name, norm_range=(0, 1), img_res=(128, 128))
    # dataloader.save_all_images(dir_path)  # (10000, 64, 64, 3)
    x = dataloader.load_datasets()  #依次导入数据
    print(x.shape, x.min(), x.max())  #尝试一次输出

    for i in range(5):
        t = time.time()
        x = dataloader.load_batch(batch_size=128)  #导入数据看下效果
        print(x.shape, x.min(), x.max())
        print(i + 1, '用时:', time.time() - t)


if __name__ == '__main__':
    main()
