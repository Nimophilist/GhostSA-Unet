# import torch
# import pickle
# import numpy as np
# import argparse
# import logging
# import torch.nn.functional as F

# from unet import UNet
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from utils.dice_score import multiclass_dice_coeff, dice_coeff


# def calculate_dsc(mask_pred, mask_true):
#     intersection = torch.sum((mask_pred == mask_true) & (mask_pred != 0), dim=(1, 2))
#     dice = (2. * intersection.float() + 1e-8) / (torch.sum(mask_pred != 0, dim=(1, 2)).float() + torch.sum(mask_true != 0, dim=(1, 2)).float() + 1e-8)
#     return dice


# # 计算Intersection over Union (IOU)
# def calculate_iou(mask_pred, mask_true):
#     intersection = torch.sum((mask_pred == mask_true) & (mask_pred != 0), dim=(1, 2))
#     union = torch.sum((mask_pred != 0) | (mask_true != 0), dim=(1, 2))
#     iou = intersection.float() / (union.float() + 1e-8)  # Add a small epsilon to avoid division by zero
#     return iou


# def calculate_accuracy(mask_pred, mask_true):
#     # Flatten the masks
#     mask_pred_flat = mask_pred.view(-1)
#     mask_true_flat = mask_true.view(-1)

#     # Calculate accuracy
#     accuracy = accuracy_score(mask_true_flat.cpu(), mask_pred_flat.cpu())

#     return accuracy

# def calculate_recall(mask_pred, mask_true):
#     # Flatten the masks
#     mask_pred_flat = mask_pred.view(-1)
#     mask_true_flat = mask_true.view(-1)

#     # Calculate recall
#     recall = recall_score(mask_true_flat.cpu(), mask_pred_flat.cpu(), average='macro', zero_division=1)

#     return recall

# def calculate_f1_score(mask_pred, mask_true):
#     # Flatten the masks
#     mask_pred_flat = mask_pred.view(-1)
#     mask_true_flat = mask_true.view(-1)

#     # Calculate F1 score
#     f1 = f1_score(mask_true_flat.cpu(), mask_pred_flat.cpu(), average='macro', zero_division=1)

#     return f1

# def calculate_precision(mask_pred, mask_true):
#     # Flatten the masks
#     mask_pred_flat = mask_pred.view(-1)
#     mask_true_flat = mask_true.view(-1)

#     # Calculate precision
#     precision = precision_score(mask_true_flat.cpu(), mask_pred_flat.cpu(), average='macro', zero_division=1)

#     return precision

# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                         help='Minimum probability value to consider a mask pixel white')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
#     return parser.parse_args()

# if __name__ == '__main__':
#     args = get_args()
#     net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model {args.model}')
#     logging.info(f'Using device {device}')

#     net.to(device=device)
#     state_dict = torch.load(args.model, map_location=device)
#     mask_values = state_dict.pop('mask_values', [0, 1])
#     net.load_state_dict(state_dict)
#     logging.info('Model loaded!')

#     with open('data_loaders.pkl', 'rb') as f:
#         data = pickle.load(f)

#     # 获取验证集数据
#     test_data = data['test']
#     # dices = []
#     # ious = []
#     # precisions = []
#     # recalls = []
#     # accuracys = []
#     # f1s = []
#     # net.eval()
#     # for i, batch in enumerate(test_data):
#     #     test_images = batch['image'].to(device)
#     #     true_test = batch['mask'].to(device)
#     #     with torch.no_grad():
#     #         output = net(test_images)
#     #         pred_test = output.argmax(dim=1)
#     #         dice = calculate_dsc(pred_test,true_test)
#     #         dices.append(dice)
#     #         iou = calculate_iou(pred_test, true_test)
#     #         ious.append(iou)
#     #         precision = calculate_precision(pred_test,true_test)
#     #         precisions.append(precision)
#     #         recall = calculate_recall(pred_test, true_test)
#     #         recalls.append(recall)
#     #         accuracy = calculate_accuracy(pred_test,true_test)
#     #         accuracys.append(accuracy)
#     #         f1 = calculate_f1_score(pred_test,true_test)
#     #         f1s.append(f1)
#     # dice = np.average([d.cpu().numpy() for d in dices])
#     # iou = np.average([iou.cpu().numpy() for iou in ious])
#     # precision = np.average([precision.cpu().numpy() for precision in precisions])
#     # recall = np.average([recall.cpu().numpy() for recall in recalls])
#     # accuracy = np.average([accuracy.cpu().numpy() for accuracy in accuracys])
#     # f1 = np.average([f1.cpu().numpy() for f1 in f1s])
    
#     # print("Dice Score:", dice)
#     # print("Iou:", iou)
#     # print("Accuracy:", accuracy)
#     # print("Recall:", recall)
#     # print("F1 Score:", f1)
#     # print("Precision:", precision)
#     # 在循环外初始化指标总和
#     dices = []
#     ious = []
#     precisions = []
#     recalls = []
#     accuracys = []
#     f1s = []
#     # total_dice = 0.0
#     # total_iou = 0.0
#     # total_precision_sum = 0
#     # total_recall_sum = 0
#     # total_accuracy_sum = 0
#     # total_f1_sum = 0
#     net.eval()
#     for i, batch in enumerate(test_data):
#         test_images = batch['image'].to(device)
#         true_tests = batch['mask'].to(device)
#         for j, (test_image, true_test) in enumerate(zip(test_images, true_tests)):
#             test_image = test_image.unsqueeze(0)
#             true_test = true_test.unsqueeze(0)
#             with torch.no_grad():
#                 output = net(test_image)
#                 pred_test = output.argmax(dim=1)
#                 # 计算当前批次的指标
#                 dice = calculate_dsc(pred_test, true_test)
#                 dices.append(dice)
#                 iou = calculate_iou(pred_test, true_test)
#                 ious.append(iou)
#                 precision = calculate_precision(pred_test, true_test)  # 添加一维以匹配真实标签的形状
#                 precisions.append(precision)
#                 recall = calculate_recall(pred_test, true_test)  # 添加一维以匹配真实标签的形状
#                 recalls.append(recall)
#                 accuracy = calculate_accuracy(pred_test, true_test)  # 添加一维以匹配真实标签的形状
#                 accuracys.append(accuracy)
#                 f1 = calculate_f1_score(pred_test, true_test)  # 添加一维以匹配真实标签的形状
#                 f1s.append(f1)
                
#                 # 将当前批次的指标总和添加到总的指标总和中
#                 # total_dice_sum += dice
#                 # total_iou_sum += iou
#                 # total_precision_sum += precision
#                 # total_recall_sum += recall
#                 # total_accuracy_sum += accuracy
#                 # total_f1_sum += f1

#     # 计算平均值
#     # avg_dice = total_dice / len(test_data)
#     # avg_iou = total_iou / len(test_data)
#     # avg_precision = total_precision_sum / len(test_data)
#     # avg_recall = total_recall_sum / len(test_data)
#     # avg_accuracy = total_accuracy_sum / len(test_data)
#     # avg_f1 = total_f1_sum / len(test_data)

#     # # 输出结果
#     # print("Average Dice Score:", avg_dice)
#     # print("Average IOU:", avg_iou)
#     # print("Average Precision:", avg_precision)
#     # print("Average Recall:", avg_recall)
#     # print("Average Accuracy:", avg_accuracy)
#     # print("Average F1 Score:", avg_f1)
# # 将张量移动到 CPU 上，并转换为 NumPy 数组
# dices_cpu = [d.cpu().numpy() for d in dices]
# ious_cpu = [i.cpu().numpy() for i in ious]


# # 计算平均值
# average_dice = np.mean(dices_cpu)
# average_iou = np.mean(ious_cpu)
# average_precision = np.mean(precisions)
# average_recall = np.mean(recalls)
# average_accuracy = np.mean(accuracys)
# average_f1 = np.mean(f1s)

# print("Average Dice Score:", average_dice)
# print("Average IOU:", average_iou)
# print("Average Precision:", average_precision)
# print("Average Recall:", average_recall)
# print("Average Accuracy:", average_accuracy)
# print("Average F1 Score:", average_f1)

import os
import torch
import pickle
import numpy as np
import argparse
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet import UNet
# from models.three_d.unet3d import UNet3D
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from data_function import MedData_test
# from hparam import hparams as hp
# from utils.dice_score import multiclass_dice_coeff, dice_coeff

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]

# source_test_dir = hp.source_test_dir
# label_test_dir = hp.label_test_dir
def calculate_dsc(mask_pred, mask_true):
    intersection = np.sum(mask_pred * mask_true, axis=(1, 2))
    dice_numerator = 2. * intersection
    total_true = np.sum(mask_true, axis=(1, 2))
    total_pred = np.sum(mask_pred, axis=(1, 2))
    smooth = 1e-6  # 平滑因子  
    # 计算Dice系数（对于每个样本）  
    dice = (dice_numerator + smooth) / (total_true + total_pred + smooth)  
    return dice


# 计算Intersection over Union (IOU)
def calculate_iou(mask_pred, mask_true):
    intersection = np.sum(mask_pred * mask_true, axis=(1, 2))  # 计算交集  
    union = np.sum(mask_pred, axis=(1, 2)) + np.sum(mask_true, axis=(1, 2)) - intersection  # 计算并集  
    smooth = 1e-6  # 平滑因子，防止除以零  
    iou = (intersection + smooth) / (union + smooth)  # 计算IoU  
    return iou
    


def calculate_accuracy(mask_pred, mask_true):
    # Flatten the masks
    mask_pred_flat = mask_pred.flatten() 
    mask_true_flat = mask_true.flatten() 

    # Calculate accuracy
    accuracy = accuracy_score(mask_true_flat, mask_pred_flat)

    return accuracy

def calculate_recall(mask_pred, mask_true):
    # Flatten the masks
    mask_pred_flat = mask_pred.flatten() 
    mask_true_flat = mask_true.flatten() 

    # Calculate recall
    recall = recall_score(mask_true_flat, mask_pred_flat, average='macro', zero_division=1)

    return recall

def calculate_f1_score(mask_pred, mask_true):
    # Flatten the masks
    mask_pred_flat = mask_pred.flatten() 
    mask_true_flat = mask_true.flatten() 

    # Calculate F1 score
    f1 = f1_score(mask_true_flat, mask_pred_flat, average='macro', zero_division=1)

    return f1

def calculate_precision(mask_pred, mask_true):
    # Flatten the masks
    mask_pred_flat = mask_pred.flatten() 
    mask_true_flat = mask_true.flatten() 

    # Calculate precision
    precision = precision_score(mask_true_flat, mask_pred_flat, average='macro', zero_division=1)

    return precision

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # net = UNet3D(in_channels=1, out_channels=args.classes, init_features=32)
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    with open('data_loaders.pkl', 'rb') as f:
        data = pickle.load(f)

    # 获取验证集数据
    test_data = data['test']

    
    dices = []
    ious = []
    precisions = []
    recalls = []
    accuracys = []
    f1s = []
    # total_dice = 0.0
    # total_iou = 0.0
    # total_precision_sum = 0
    # total_recall_sum = 0
    # total_accuracy_sum = 0
    # total_f1_sum = 0
    net.eval()
    for i, batch in enumerate(test_data):
        test_image = batch['image'].to(device)
        true_test = batch['mask'].to(device)
    #     for j, (test_image, true_test) in enumerate(zip(test_images, true_tests)):
    #         test_image = test_image.unsqueeze(0)
    #         true_test = true_test.unsqueeze(0)
    # for i, batch in enumerate(test_loader):
    #     # test_images = batch['image'].to(device)
    #     # true_tests = batch['mask'].to(device)
    #     test_image = batch['source']['data'].type(torch.FloatTensor).cuda()
    #     true_test = batch['label']['data'].type(torch.FloatTensor).cuda()
    #     true_test = np.squeeze(true_test, axis=1)
        # print("gt:",true_test.shape)
        # print("img:",test_image.shape)
        # for j, (test_image, true_test) in enumerate(zip(test_images, true_tests)):
        #     test_image = test_image.unsqueeze(0)
        #     true_test = true_test.unsqueeze(0)
        with torch.no_grad():
            output = net(test_image)
            pred_test = output.argmax(dim=1).cpu().numpy()
            true_test_cpu = true_test.cpu().numpy()
            # print("true:",true_test_cpu.shape)
            # print("pred:",pred_test.shape)
            # print(true_test.shape)
            # 计算当前批次的指标
            dice_scores_cls = []
            iou_cls = []
            precision_cls = []
            recall_cls = []
            accuracy_cls = []
            f1_cls = []

            for cls in range(args.classes):  
                if true_test_cpu.ndim == 3:  # 假设是(batch, height, width)的整数编码  
                    true_test_cls = (true_test_cpu == cls).astype(np.float32)  
                else:  # 如果是one-hot编码，则直接取对应类别的通道  
                    true_test_cls = true_test_cpu[:, cls, ...].cpu().numpy()  
                # print(true_test_cls.shape)
                # 获取当前类别的预测概率图  
                pred_test_cls = (pred_test == cls).astype(np.float32)  
                # print(pred_test_cls.shape)
                # 计算当前类别的Dice系数  
                dice = calculate_dsc(pred_test_cls, true_test_cls)  # 假设calculate_dsc是一个自定义函数  
                dice_scores_cls.append(dice)
                iou = calculate_iou(pred_test_cls, true_test_cls)
                iou_cls.append(iou)
                precision = calculate_precision(pred_test_cls, true_test_cls)  # 添加一维以匹配真实标签的形状
                precision_cls.append(precision)
                recall = calculate_recall(pred_test_cls, true_test_cls)  # 添加一维以匹配真实标签的形状
                recall_cls.append(recall)
                accuracy = calculate_accuracy(pred_test_cls, true_test_cls)  # 添加一维以匹配真实标签的形状
                accuracy_cls.append(accuracy)
                f1 = calculate_f1_score(pred_test_cls, true_test_cls)  # 添加一维以匹配真实标签的形状
                f1_cls.append(f1)

            dice_score = np.mean(dice_scores_cls)  
            dices.append(dice_score)

            iou_avg = np.mean(iou_cls)
            ious.append(iou_avg)

            precision_avg = np.mean(precision_cls)
            precisions.append(precision_avg)
            
            recall_avg = np.mean(recall_cls)
            recalls.append(recall_avg)

            accuracy_avg = np.mean(accuracy_cls)
            accuracys.append(accuracy_avg)
           
            f1_avg = np.mean(f1_cls)
            f1s.append(f1_avg)
                
               


# 计算平均值
average_dice = np.mean(dices)
average_iou = np.mean(ious)
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_accuracy = np.mean(accuracys)
average_f1 = np.mean(f1s)

print("Average Dice Score:", average_dice)
print("Average IOU:", average_iou)
print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average Accuracy:", average_accuracy)
print("Average F1 Score:", average_f1)