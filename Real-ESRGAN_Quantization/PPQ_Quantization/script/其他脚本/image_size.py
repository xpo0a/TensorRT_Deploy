import os.path
import glob
import cv2

def main():
    minH = 9999
    minW = 9999
    maxHeight = 0
    maxWeight = 0
    for file in glob.glob(r'/home/ubuntu/Music/TRT_data/INT8_data/*.png'):
        src = cv2.imread(file, cv2.IMREAD_ANYCOLOR)
        maxHeight = max(src.shape[0], maxHeight)
        maxWeight = max(src.shape[1], maxWeight)
        minH, minW = min(src.shape[0], minH), min(src.shape[1], minW)
    print(maxHeight, maxWeight)
    print(minH, minW)
if __name__ == '__main__':
    main()
