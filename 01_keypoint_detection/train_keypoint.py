import argparse
import cv2
import os
import csv
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
try:
    import tensorflow as tf
except ImportError:
    class _Writer:
        def as_default(self):
            return nullcontext()

    class _Summary:
        @staticmethod
        def create_file_writer(_):
            return _Writer()

        @staticmethod
        def scalar(*args, **kwargs):
            pass

    class _TensorFlowFallback:
        summary = _Summary()

    tf = _TensorFlowFallback()
from datasets.coco_new import CocoTrainDataset
from datasets.transformations import ConvertKeypoints, Scale, Rotate, CropPad, Flip
from modules.get_parameters import get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.loss import l2_loss
from modules.load_state import load_state, load_from_mobilenet
from val import evaluate

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def visualize_feature_maps(feature_maps,save_path):
    # feature_maps =feature_maps.transpose((1,2,0))
    num_feature_maps = feature_maps.shape[2]  # Number of feature maps.
    num_cols = 3  # Number of feature maps per row.
    num_rows = num_feature_maps // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    # fig, axes = plt.subplots(1, 3)

    for i in range(num_feature_maps):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        ax.imshow(feature_maps[:,:,i], cmap='viridis')
        ax.axis('off')

    # plt.show()
    plt.savefig(save_path)

def train(prepared_train_labels, train_images_folder, num_refinement_stages, base_lr, batch_size, batches_per_iter,
          num_workers, checkpoint_path, weights_only, from_mobilenet, checkpoints_folder, log_after,
          val_labels, val_images_folder, val_output_name, checkpoint_after, val_after):
    net = PoseEstimationWithMobileNet(num_refinement_stages)
    # stage1_pafs_loss_all = []
    # stage1_heatmaps_loss_all = []
    # stage2_pafs_loss_all = []
    # stage2_heatmaps_loss_all = []
    # stride = 8
    # sigma = 7
    stride = 8
    sigma = 5
    path_thickness = 1
    dataset = CocoTrainDataset(prepared_train_labels, train_images_folder,
                               stride, sigma, path_thickness,
                               transform=transforms.Compose([
                                   Scale(),
                                   Rotate(pad=(255, 255, 255)),
                                   CropPad(pad=(255, 255, 255)),  # Pad with a white background.
                                   # Rotate(pad=(128, 128, 128)),
                                   # CropPad(pad=(128, 128, 128)),
                                   ]))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    optimizer = optim.Adam([
        {'params': get_parameters_conv(net.model, 'weight')},
        {'params': get_parameters_conv_depthwise(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(net.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(net.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(net.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(net.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(net.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(net.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)

    num_iter = 0
    current_epoch = 0
    drop_after_epoch = [100, 200, 260]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)

        if from_mobilenet:
            load_from_mobilenet(net, checkpoint)
        else:
            load_state(net, checkpoint)
            if not weights_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                num_iter = checkpoint['iter']
                current_epoch = checkpoint['current_epoch']

    net = DataParallel(net).cuda()
    net.train()

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    train_log_dir_stage1 = 'logs/train/stage1/' + TIMESTAMP
    train_log_dir_stage2 = 'logs/train/stage2/' + TIMESTAMP
    writer_train1 = tf.summary.create_file_writer(train_log_dir_stage1)
    writer_train2 = tf.summary.create_file_writer(train_log_dir_stage2)

    for epochId in range(current_epoch, 5000):
        print("\n ##########################################################")
        print("Epoch = " + str(epochId))
        scheduler.step()
        total_losses = [0] * (num_refinement_stages + 1) # heatmaps loss, paf loss per stage
        batch_per_iter_idx = 0
        for batch_data in train_loader:
            if batch_per_iter_idx == 0:
                optimizer.zero_grad()

            images = batch_data['image'].cuda()
            keypoint_masks = batch_data['keypoint_mask'].cuda()
            keypoint_maps = batch_data['keypoint_maps'].cuda()

            stages_output = net(images)

            losses = []
            for loss_idx in range(len(total_losses)):
                losses.append(l2_loss(stages_output[loss_idx], keypoint_maps, 1 - keypoint_masks, images.shape[0]))
                total_losses[loss_idx] += losses[0].item() / batches_per_iter

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]
            loss /= batches_per_iter
            loss.backward()
            batch_per_iter_idx += 1
            if batch_per_iter_idx == batches_per_iter:
                optimizer.step()
                batch_per_iter_idx = 0
                num_iter += 1
            else:
                continue

            if num_iter % log_after == 0:
                print('Iter: {}'.format(num_iter))
                for loss_idx in range(len(total_losses)):
                    print('\n'.join(['stage{}_heatmaps_loss: {}']).format(

                        loss_idx + 1, total_losses[loss_idx] / log_after))
                    if loss_idx + 1 == 1:
                        l = []
                        l.append(total_losses[loss_idx ] / log_after)
                        with open('stage1.csv', 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(l)
                    
                        with writer_train1.as_default():
                            tf.summary.scalar('stage1', total_losses[0 ] / log_after,epochId)
                    
                    else:
                        l = []
                        l.append(total_losses[loss_idx ] / log_after)
                        with open('stage2.csv', 'a', newline='') as f1:
                            writer = csv.writer(f1)
                            writer.writerow(l)
                        with writer_train2.as_default():
                            tf.summary.scalar('stage2', total_losses[1 ] / log_after,epochId)

                for loss_idx in range(len(total_losses)):
                    total_losses[loss_idx] = 0
            if num_iter % checkpoint_after == 0:
                snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
                torch.save({'state_dict': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iter': num_iter,
                            'current_epoch': epochId},
                           snapshot_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-train-labels', type=str, required=True,
                        help='path to the file with prepared annotations')
    parser.add_argument('--train-images-folder', type=str, required=True, help='path to COCO train images folder')
    parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')
    parser.add_argument('--base-lr', type=float, default=9e-4, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--batches-per-iter', type=int, default=1, help='number of batches to accumulate gradient from')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--checkpoint-path', type=str, default='',
                        help='optional checkpoint used to initialize or resume training')
    parser.add_argument('--from-mobilenet', action='store_true',
                        help='load weights from mobilenet feature extractor')
    parser.add_argument('--weights-only', action='store_true',
                        help='just initialize layers with pre-trained weights and start training from the beginning')
    parser.add_argument('--experiment-name', type=str, default='default_0928',
                        help='experiment name to create folder for checkpoints')
    parser.add_argument('--log-after', type=int, default=1, help='number of iterations to print train loss')

    parser.add_argument('--val-labels', type=str, default='', help='optional validation annotation JSON')
    parser.add_argument('--val-images-folder', type=str, default='', help='optional validation image folder')
    parser.add_argument('--val-output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--checkpoint-after', type=int, default=1000,
                        help='number of iterations to save checkpoint')
    parser.add_argument('--val-after', type=int, default=1,
                        help='number of iterations to run validation')
    args = parser.parse_args()

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(args.prepared_train_labels, args.train_images_folder, args.num_refinement_stages, args.base_lr, args.batch_size,
          args.batches_per_iter, args.num_workers, args.checkpoint_path, args.weights_only, args.from_mobilenet,
          checkpoints_folder, args.log_after, args.val_labels, args.val_images_folder, args.val_output_name,
          args.checkpoint_after, args.val_after)
