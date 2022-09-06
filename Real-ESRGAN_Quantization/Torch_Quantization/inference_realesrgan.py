import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
import time
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from torchinfo import summary
# from script.calculatePSNR_SSIM import Calculate

model_path = "/home/ubuntu/Music/Real-ESRGAN-QAT_torch/experiments/model_saved/net_g_140000.pth"

def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/ubuntu/Music/Real-ESRGAN-QAT/inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus'
              'RealESRGANv2-anime-xsx2 | RealESRGANv2-animevideo-xsx2-nousm | RealESRGANv2-animevideo-xsx2'
              'RealESRGANv2-anime-xsx4 | RealESRGANv2-animevideo-xsx4-nousm | RealESRGANv2-animevideo-xsx4'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in [
            'RealESRGANv2-anime-xsx2', 'RealESRGANv2-animevideo-xsx2-nousm', 'RealESRGANv2-animevideo-xsx2'
    ]:  # x2 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
    elif args.model_name in [
            'RealESRGANv2-anime-xsx4', 'RealESRGANv2-animevideo-xsx4-nousm', 'RealESRGANv2-animevideo-xsx4'
    ]:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths

    # summary(model, (8, 3, 500, 400))

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    all_time_start = time.time()
    MaxTime = ['0', -99999]  # index, time
    MinTime = ['0', 999999]
    num = 0

    Time_full = '/home/ubuntu/Music/Real-ESRGAN-master/TXT_File/time.txt'
    PSNR_full = '/home/ubuntu/Music/Real-ESRGAN-master/TXT_File/psnr.txt'
    SSIM_full = '/home/ubuntu/Music/Real-ESRGAN-master/TXT_File/ssim.txt'
    file_time = open(Time_full, 'a+')
    file_psnr = open(PSNR_full, 'a+')
    file_ssim = open(SSIM_full, 'a+')
    Dur = []

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if True:
                # warm-up
                num += 1
                print(111111111111111111)
                output, _ = upsampler.enhance(img, outscale=args.outscale)
                print(111111111111111111)

                dur_time = 0
                iters = 1
                for _ in range(iters):
                    begin = time.time()
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
                    dur = time.time() - begin
                    if dur > MaxTime[1]:
                        MaxTime[0], MaxTime[1] = num,dur
                    if dur < MinTime[1]:
                        MinTime[0], MinTime[1] = num, dur
                    print('{} dur time : {}'.format(idx, dur))
                Dur.append(dur)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            save_path = os.path.join(args.output, f'{imgname}{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)
    all_time_dur = time.time() - all_time_start

    # HR = '/home/ubuntu/Music/TRT_data/DIV2KRK_public/DIV2KRK/gt/'  # / 不能丢
    # SR = '/home/ubuntu/Music/Real-ESRGAN-master/results/'
    #
    #
    # point = 4
    # MaxTime[1]=round(MaxTime[1], point)
    # MinTime[1]=round(MinTime[1], point)
    # all_time_dur=round(all_time_dur, point)
    # all_time_dur=round(all_time_dur,point)
    # perTime = round((all_time_dur / 5), point)
    # psrn_list, ssim_list = Calculate(HR, SR, 5)
    #
    # for i in range(5):
    #     file_psnr.writelines([str(psrn_list[i]), '\n'])
    #     file_time.writelines([str(Dur[i]), '\n'])
    #     file_ssim.writelines([str(ssim_list[i]), '\n'])
    # file_psnr.close()
    # file_ssim.close()
    # file_time.close()
    #
    # average_psnr = round(sum(psrn_list) / len(psrn_list), point)
    # average_ssim = round(sum(ssim_list) / len(ssim_list), point)
    # print(psrn_list)
    #
    # print('------------------- summary ----------------------')
    # print('Pytorch infer')
    # print('about Time, Test Data is DIV2KRK (gt, lrx4) x 100:')
    # print('The Max infer time = {} s, pic idx = {}'.format(MaxTime[1], MaxTime[0]))
    # print('The Min infer time = {} s, img idx = {}'.format(MinTime[1], MinTime[0]))
    # print('{} pictures inference time = {} s'.format(idx, all_time_dur))
    # print('The average time of infer one picture = {} s'.format(perTime))
    # print('*************************************************')
    # print('PSNR and SSIM')
    # print('The average PSNR = {}'.format(average_psnr))
    # print('The average SSIM = {}'.format(average_ssim))


if __name__ == '__main__':
    main()
