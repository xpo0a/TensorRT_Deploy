import os
import matplotlib.pyplot as plt

x1, y1 = [], []  # FP32
x2, y2 = [], []  # FP16
x3, y3 = [], []  # Pytorch
x4, y4 = [], []  # int8
str1 = '/home/ubuntu/Music/Real-ESRGAN_py/TXT_File/FP32/time.txt'
str2 = '/home/ubuntu/Music/Real-ESRGAN_py/TXT_File/FP16/time.txt'
str3 = '/home/ubuntu/Music/Real-ESRGAN_py/TXT_File/pytorch/time.txt'
str4 = '/home/ubuntu/Music/Real-ESRGAN_py/TXT_File/int8/time.txt'

with open(str1, 'r+') as f1:
    content1 = f1.read()

a1 = content1.split('\n')
print(len(a1))
for i in range(len(a1) - 1):
    y1.append(float(a1[i]))
    x1.append(i)

with open(str2, 'r+') as f2:
    content2 = f2.read()

a2 = content2.split('\n')
print(len(a2))
for i in range(len(a2) - 1):
    y2.append(float(a2[i]))

with open(str3, 'r+') as f3:
    content3 = f3.read()

a3 = content3.split('\n')
print(len(a3))
for i in range(len(a3) - 1):
    y3.append(float(a3[i]))

with open(str4, 'r+') as f4:
    content4 = f4.read()

a4 = content4.split('\n')
print(len(a4))
for i in range(len(a4) - 1):
    y4.append(float(a4[i]))

fig = plt.figure()
ax1=plt.subplot(2,2,1)
plt.plot(x1, y1, label='TRT-FP32', color='r', marker='^', linestyle='dashed')
# plt.axis([0, 110, 0.2, 2])
plt.legend()

ax2=plt.subplot(2,2,2)
plt.plot(x1, y2, label='TRT-FP16', color='b', marker='v', linestyle='dashed')
plt.legend()

ax3=plt.subplot(2,2,3)
plt.plot(x1, y3, label='pytorch', color='g', marker='*', linestyle='dashed')
plt.legend()

ax2=plt.subplot(2,2,4)
# plt.plot(x1, y1, label='TRT-FP32', color='r', marker='^', linestyle='dashed')
# plt.plot(x1, y2, label='TRT-FP16', color='b', marker='v', linestyle='dashed')
# plt.plot(x1, y3, label='pytorch-FP32', color='g', marker='*', linestyle='dashed')
plt.plot(x1, y4, label='TRT-INT8', color='y', marker='+', linestyle='dashed')
plt.legend()

plt.axis([0, 110, 0.2, 2])
plt.xlabel('img index')
# y轴文本
plt.ylabel('time/s')
plt.title('')
plt.show()
fig.savefig('inferTime_all.png')
