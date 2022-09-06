import os
import glob
import cv2
from PIL import Image

ImgDir = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/gt'
SaveFIle = "/home/ubuntu/Music/Real-ESRGAN_PPQ/working/data/"
SaveFile_256 = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/gt_resize/'

if os.path.isfile(ImgDir):
    paths = [ImgDir]
else:
    paths = sorted(glob.glob(os.path.join(ImgDir, '*')))

for i, path in enumerate(paths):
    imgname, extension = os.path.splitext(os.path.basename(path))
    print('Saving', i, imgname)

    # img1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # img = np.resize(img1, (256, 256, 3))
    # np.save(file=f'{SaveFIle}{i+1}', arr=img)

    img = Image.open(path)
    crop_img = img.crop((0, 0, 1024, 1024))
    crop_img.save(f'{SaveFile_256}{i+1}.png')