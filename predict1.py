import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import pickle
from PIL import Image
from torchvision import transforms
from pathlib import Path

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from natsort import natsorted
from os.path import isfile,join

def predict_img(net,
                full_img,
                device):
    net.eval()
    # img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    # image = []
    # for img in full_img:
    #         img_array = np.array(img)  # 将PIL图像转换为NumPy数组
    #         if (img_array > 1).any():
    #             img_array = img_array / 255.0  
    #         image.append(img_array)
    # stacked_images = np.stack(image, axis=0)
    # img = torch.as_tensor(stacked_images).float().contiguous()
    # print(img.shape)
    # img = img.unsqueeze(0)
    # img = img.to(device=device, dtype=torch.float32)
    img = torch.as_tensor(full_img).float().contiguous()
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        print(output.shape)
        output = F.interpolate(output, (512, 512), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask.long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=12, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    if args.output:
        return args.output
    else:
        return list(map(_generate_name, [file for file in os.listdir(args.input)]))



def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1], 3), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    if mask_values == [0, 1]:
        for i, v in enumerate(mask_values):
            out[mask == i] = v
        return Image.fromarray(out)
    else:
        # for i, _ in enumerate(mask_values):
        #     out[mask == i] = list(np.random.choice(range(256), size=3))
        for i, color in mask_values.items():
            out[mask == i] = color  # 使用颜色映射为每个类别分配颜色
        return Image.fromarray(out, "RGB")


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = Path(args.input)
    out_files = args.output
    # with open('data_loaders.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # # 获取验证集数据
    # test_data = data['test']
    # test_images = test_data['images']
    


    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    mask_values = {
    0: (0, 0, 0),    # 类别0的颜色为黑色
    1: (128, 174, 128),  # 类别1的颜色为#80ae80的RGB值(128, 174, 128)
    2: (241, 214, 145),  # 类别2的颜色为#f1d691的RGB值
    3: (177, 122, 101),
    4: (111, 184, 210),
    5: (216, 101, 79),
    6: (221, 130, 101),
    7: (144, 238, 144),
    8: (192, 104, 88),
    9: (220, 245, 20),
    10: (78, 63, 0),
    11: (255, 250, 220),
    # 12: (230, 220, 70),
    # 13: (200, 200, 235),
    # 14: (250, 250, 210),
    # 15: (244, 214, 49),
    # 16: (0, 151, 206),
    # 17: (216, 101, 79),
    # 18: (183, 156, 220),
    # 19: (183, 214, 211),
}
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    file_prefixes = set(file.split('_')[0] for file in os.listdir(in_files) if isfile(join(in_files, file)))
    for prefix in file_prefixes:
        img_files = natsorted([file for file in os.listdir(in_files) if file.startswith(prefix)])
        if len(img_files) < 3:
            logging.warning(f'Not enough images with prefix {prefix} to form a three-channel image.')
            continue
        images = [Image.open(os.path.join(in_files, file)) for file in img_files[:3]]  # 取前三个图像
        image = [np.array(img) / 255.0 for img in images]
        stacked_images = np.stack(image, axis=0)
    
        if not args.no_save:
            masks = predict_img(net=net,full_img=stacked_images,device=device)
            result = mask_to_image(masks, mask_values)
            result.save(os.path.join(out_files, img_files[1]))
            logging.info(f'Mask saved to {img_files[1]}')
        
        if args.viz:
                    logging.info(f'Visualizing results for image {filename}, close to continue...')
                    plot_img_and_mask(img, mask)
    # print(len(masks))
    # for i, filename in enumerate(in_files):
    #     logging.info(f'Predicting image {filename} ...')
    #     img = Image.open(filename)
    #
    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)
    # for i, batch in enumerate(test_data):
    #     test_images = batch['image'].to(device)
    #     for j, test in enumerate(test_images):
    #         test = test.unsqueeze(0)
    #         if not args.no_save:
    #             out_filename = f'batch_{i}_{j}_OutPut.png'
    #             masks = predict_img(net=net,full_img=test,device=device)
    #             result = mask_to_image(masks, mask_values)
    #             result.save(os.path.join(out_files, out_filename))
    #             logging.info(f'Mask saved to {out_filename}')

    #         if args.viz:
    #             logging.info(f'Visualizing results for image {filename}, close to continue...')
    #             plot_img_and_mask(img, mask)
