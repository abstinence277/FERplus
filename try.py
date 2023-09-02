import cv2
import os
import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import csv
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
   # 图像个数

# 遍历每张图片
#新的数据集命名为了fer2021,分别打开每个数据集进行编写，首先training部分:

with open("data2/fer20220121.csv", 'w',newline='') as f:
    f.write('emotion,pixels,Usage\n')
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/Training/Anger/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i],cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        data = cv2.resize(img_ndarray,(48,48))#图像大小48*48像素值
        data = data.reshape(-1)
        s=data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['0',pixels,"Training"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/Training/Disgust/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray,(48, 48))
        data = data.reshape(-1)
        csv_writer = csv.writer(f)
        s = data.shape
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['1', pixels, "Training"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/Training/Fear/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['2', pixels, "Training"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/Training/Happy/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['3', pixels, "Training"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/Training/Sadness/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['4', pixels, "Training"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/Training/Surprise/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['5', pixels, "Training"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/Training/Neutral/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['6', pixels, "Training"])


    #PrivateTest部分：
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PrivateTest/Anger/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['0', pixels, "PrivateTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PrivateTest/Disgust/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['1', pixels, "PrivateTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PrivateTest/Fear/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['2', pixels, "PrivateTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PrivateTest/Happy/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        csv_writer = csv.writer(f)
        s = data.shape
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['3', pixels, "PrivateTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PrivateTest/Sadness/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['4', pixels, "PrivateTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PrivateTest/Surprise/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['5', pixels, "PrivateTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PrivateTest/Neutral/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['6', pixels, "PrivateTest"])


#publicTest部分：
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PublicTest/Anger/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        # print(pixels)
        # 将结果写入 csv
        csv_writer.writerow(['0', pixels, "PublicTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PublicTest/Disgust/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['1', pixels, "PublicTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PublicTest/Fear/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['2', pixels, "PublicTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PublicTest/Happy/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['3', pixels, "PublicTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PublicTest/Sadness/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['4', pixels, "PublicTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PublicTest/Surprise/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['5', pixels, "PublicTest"])
    c = get_imlist(r"E:/desktop/Process/Facial-Expression-Recognition.Pytorch-master/FER2013 plus/data2/data/data/PublicTest/Neutral/")
    d = len(c)
    for i in range(d):
        img = cv2.imread(c[i], cv2.IMREAD_GRAYSCALE)  # 打开图像
        img_ndarray = np.asarray(img)  # 将图像转化为数组并将像素转化到0-1之间
        # noinspection PyRedeclaration
        data = cv2.resize(img_ndarray, (48, 48))
        data = data.reshape(-1)
        s = data.shape
        csv_writer = csv.writer(f)
        pixels = (" ".join(str(s) for s in data))
        csv_writer.writerow(['6', pixels, "PublicTest"])
f.close()
