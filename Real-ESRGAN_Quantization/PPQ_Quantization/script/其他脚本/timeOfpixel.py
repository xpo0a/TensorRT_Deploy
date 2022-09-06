import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    LR = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/lr_x4/'
    img_size = Calculate(LR, 100)
    str1 = '/home/ubuntu/Music/Real-ESRGAN_py/TXT_File/FP32/time.txt'

    with open(str1, 'r+') as f1:
        content1 = f1.read()
    y1 = [] # time
    dic = {} # time: pixel
    a1 = content1.split('\n')
    for i in range(len(a1) - 1):
        y1.append(float(a1[i]))

    for i in range(100):
        dic[y1[i]] = img_size[i]

    Dic = sorted(dic.items(), key=lambda x:x[1])
    print(Dic)
    x, y = [], []
    for i in Dic:
        x.append(i[0])
        y.append(i[1])
    print(x, y)

    fig = plt.figure()
    plt.plot(y, x, label='FP32', color='r', marker='^', linestyle='dashed')
    plt.ylabel('infer time/s')
    plt.xlabel('number of pixel')
    plt.show()
    fig.savefig('TimeOfPixel.png')



def Calculate(HRDir, picNum):
    HRList = os.listdir(HRDir)

    # pic size
    i, c = 0, 0
    img_size = []
    for i in range(1, picNum + 1):
        gt = HRDir + str(i) + '.png'
        img_gt = cv2.imread(gt, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        shape = img_gt.shape
        img_size.append(shape[1] * shape[0])
        # img_size.append(img_gt.shape)
    return img_size

# x1, y1 = [], []  # FP32
# x2, y2 = [], []  # FP16
# x3, y3 = [], []  # Pytorch
# str1 = '/home/ubuntu/Music/Real-ESRGAN_py/TXT_File/FP32/time.txt'
# str2 = '/home/ubuntu/Music/Real-ESRGAN_py/TXT_File/FP16/time.txt'
# str3 = '/home/ubuntu/Music/Real-ESRGAN_py/TXT_File/pytorch/time.txt'
#
# with open(str1, 'r+') as f1:
#     content1 = f1.read()
#
# a1 = content1.split('\n')
# print(len(a1))
# for i in range(len(a1) - 1):
#     y1.append(float(a1[i]))
#     x1.append(i)
#
# with open(str2, 'r+') as f2:
#     content2 = f2.read()
#
# a2 = content2.split('\n')
# print(len(a2))
# for i in range(len(a2) - 1):
#     y2.append(float(a2[i]))
#
# with open(str3, 'r+') as f3:
#     content3 = f3.read()
#
# a3 = content3.split('\n')
# print(len(a3))
# for i in range(len(a3) - 1):
#     y3.append(float(a3[i]))
#
# fig = plt.figure()
# plt.plot(x1, y1, label='FP32', color='r', marker='^', linestyle='dashed')
# plt.plot(x1, y2, label='FP16', color='b', marker='v', linestyle='dashed')
# plt.plot(x1, y3, label='pytorch', color='g', marker='*', linestyle='dashed')
#
# plt.legend()
#
# plt.axis([0, 110, 0.2, 2])
# plt.xlabel('img index')
# # y轴文本
# plt.ylabel('time/s')
# plt.title('img inference time')
# plt.show()
# fig.savefig('time.png')

if __name__ == '__main__':
    main()