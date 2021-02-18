import logging
import os
import shutil
import math
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms
import PIL

from augmentation import RandAugment
from regularization import (SmoothCrossEntropy, GeneralizedCrossEntropy,
                            mixup_data, mixup_criterion,
                            cutmix_data, cutmix_criterion)

logger = logging.getLogger(__name__)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, bs=1, gs=1):
        self.val = val
        self.sum += val * bs * gs
        self.count += bs
        self.avg = self.sum / self.count


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class TransformSSL(object):
    def __init__(self, args, mean, std, interpolation):
        n, m = args.randaug
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(args.resize, scale=(
                args.imagenet_crop, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            RandAugment(n=n, m=m),
            transforms.RandomResizedCrop(args.resize, scale=(
                args.imagenet_crop, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def accuracy(path, output, target, topk=(1,)):
    with torch.no_grad():
        output = output.to(torch.device('cpu'))
        target = target.to(torch.device('cpu'))
        maxk = max(topk)
        batch_size = target.shape[0]

        _, idx = output.sort(dim=1, descending=True)
        pred = idx.narrow(1, 0, maxk).t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def get_image_processing(args):
    if 'ViT' in args.arch:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        interp = 2
        resample_mode = PIL.Image.BILINEAR
    elif 'BiT' in args.arch:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        interp = 2
        resample_mode = PIL.Image.BILINEAR
    elif 'efficientnet' in args.arch:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        interp = 3
        resample_mode = PIL.Image.BICUBIC
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interp = 2
        resample_mode = PIL.Image.BILINEAR
    return mean, std, interp, resample_mode


def get_transform(args):
    mean, std, interpolation, resample_mode = get_image_processing(args)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if args.imagenet_crop > 0:
        train_transform.transforms.insert(
            0, transforms.RandomResizedCrop(
                args.resize, scale=(args.imagenet_crop, 1.0), interpolation=interpolation))
    else:
        train_transform.transforms.insert(
            0, transforms.Resize(int(args.resize), interpolation=interpolation))
        train_transform.transforms.insert(1, transforms.RandomCrop(args.resize))

    val_transform = transforms.Compose([
        transforms.Resize(int(args.resize/0.875), interpolation=interpolation),
        # transforms.Resize(int(args.resize), interpolation=interpolation),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        normalize,
    ])
    if args.jitter > 0:
        train_transform.transforms.insert(1, transforms.ColorJitter(args.jitter,
                                                                    args.jitter,
                                                                    args.jitter,
                                                                    0.1))

    if args.randaug is not None:
        n, m = args.randaug
        train_transform.transforms.insert(0, RandAugment(n, m, resample_mode))

    return train_transform, val_transform


def get_transform_ssl(args):
    mean, std, interpolation, resample_mode = get_image_processing(args)
    normalize = transforms.Normalize(mean=mean, std=std)
    val_transform = transforms.Compose([
        transforms.Resize(int(args.resize/0.875), interpolation=interpolation),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        normalize,
    ])
    labeled_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if args.imagenet_crop:
        labeled_transform.transforms.insert(
            0, transforms.RandomResizedCrop(
                args.resize, scale=(args.imagenet_crop, 1.0), interpolation=interpolation))
    else:
        labeled_transform.transforms.insert(
            0, transforms.Resize(int(args.resize), interpolation=interpolation))
        labeled_transform.transforms.insert(1, transforms.RandomCrop(args.resize))

    if args.jitter > 0:
        labeled_transform.transforms.insert(1, transforms.ColorJitter(args.jitter,
                                                                      args.jitter,
                                                                      args.jitter,
                                                                      0.1))

    unlabeled_transform = TransformSSL(args, mean=mean, std=std, interpolation=interpolation)

    return labeled_transform, unlabeled_transform, val_transform


def save_checkpoint(args, state, is_best):
    os.makedirs(args.save_path, exist_ok=True)
    filename = f'{args.save_path}/{args.name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.save_path}/{args.name}_best.pth.tar')


def create_loss_fn(args):
    if args.label_smoothing > 0:
        criterion = SmoothCrossEntropy(alpha=args.label_smoothing)
    elif args.sigmoid:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion.to(args.device)


def compute_loss_with_regularization(args, model, criterion, images, targets, train=True):
    if train:
        if args.mixup > 0 and args.cutmix > 0:
            if random.random() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, targets, args.mixup, args)
                output = model(images)
                if args.sigmoid:
                    targets_a = F.one_hot(targets_a, num_classes=args.num_classes).float()
                    targets_b = F.one_hot(targets_b, num_classes=args.num_classes).float()
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)

            else:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, args)
                output = model(images)
                if args.sigmoid:
                    targets_a = F.one_hot(targets_a, num_classes=args.num_classes).float()
                    targets_b = F.one_hot(targets_b, num_classes=args.num_classes).float()
                loss = cutmix_criterion(criterion, output, targets_a, targets_b, lam)

        elif args.mixup > 0:
            images, target_a, target_b, lam = mixup_data(images, targets, args.mixup, args)
            output = model(images)
            if args.sigmoid:
                targets_a = F.one_hot(targets_a, num_classes=args.num_classes).float()
                targets_b = F.one_hot(targets_b, num_classes=args.num_classes).float()
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)

        elif random.random() < args.cutmix:
            images, targets_a, targets_b, lam = cutmix_data(images, targets, args)
            output = model(images)
            if args.sigmoid:
                targets_a = F.one_hot(targets_a, num_classes=args.num_classes).float()
                targets_b = F.one_hot(targets_b, num_classes=args.num_classes).float()
            loss = cutmix_criterion(criterion, output, targets_a, targets_b, lam)

        else:
            output = model(images)
            if args.sigmoid:
                F.one_hot(targets, num_classes=args.num_classes)
            loss = criterion(output, targets)
        return loss

    else:
        output = model(images)
        if args.sigmoid:
            F.one_hot(targets, num_classes=args.num_classes)
            loss = criterion(output, targets)
        else:
            loss = F.cross_entropy(output, targets)
        return loss, output


def compute_loss_ssl(args, model, criterion, images_l, images_uw, images_us, targets):
    batch_size = images_l.shape[0]
    images = torch.cat((images_l, images_uw, images_us))
    logits = model(images)
    logits_l = logits[:batch_size]
    logits_uw, logits_us = logits[batch_size:].chunk(2)
    del logits

    loss_l = criterion(logits_l, targets)

    soft_pseudo_label = torch.softmax(logits_uw.detach()/args.temperature, dim=-1)
    max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold).float()
    # loss_u = (F.cross_entropy(logits_us, hard_pseudo_label, reduction='none') * mask).mean()
    loss_u = (-(soft_pseudo_label * torch.log_softmax(logits_us, dim=-1)).sum(dim=-1) * mask).mean()
    if args.warmup_u > 0:
        weight_u = args.lambda_u * min(1., (args.global_step + 1) / args.warmup_u)
    else:
        weight_u = args.lambda_u
    loss = loss_l + weight_u * loss_u

    return loss, loss_l, loss_u, mask.mean()


def module_load_state_dict(model, state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
