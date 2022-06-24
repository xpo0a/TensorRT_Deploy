import cv2
import numpy as np
from os import path as osp
import os
import glob


def Rename():
    picNum = 100
    HRDir = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/gt'
    SRDir = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/lr_x4'
    HRList = os.listdir(HRDir)
    SRList = os.listdir(SRDir)
    if len(HRList) != picNum or len(SRList) != picNum:
        print('number of HR image != SR')
        return

    # pic rename
    i, c = 0, 0
    for item in HRList:
        if item.endswith('.png'):
            src = os.path.join(os.path.abspath(HRDir), item)
            k = item.split('_')[1]
            print(k)
            dst = os.path.join(os.path.abspath(HRDir), str(k) + '.png')
            try:
                os.rename(src, dst)
                i += 1
                c += 1
            except:
                continue
    i, c = 0, 0
    for item in SRList:
        if item.endswith('.png'):
            src = os.path.join(os.path.abspath(SRDir), item)
            k = item.split('_')[1] # 2.png
            dst = os.path.join(os.path.abspath(SRDir), str(k))

            try:
                os.rename(src, dst)
                i += 1
                c += 1
            except:
                continue

def main():
    Rename()

if __name__ == '__main__':
    main()