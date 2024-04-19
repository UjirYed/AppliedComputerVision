import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
import glob
import itertools
from PIL import Image
from PIL.Image import BILINEAR
from torchinfo import summary
from transformers import (
    TrainingArguments,
    Trainer,
    EfficientNetConfig,
    EfficientNetForImageClassification,
    ResNetForImageClassification,
    ResNetConfig
)
import evaluate
import accelerate
import tqdm


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x[0] for x in batch]),
        "labels": torch.LongTensor([int(x[1]) for x in batch]),
    }


if __name__ == "__main__":
    # Create the data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the dataset
    data_dir = "../data/dataset/"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    
    resnet_custom_config = ResNetConfig(
        embedding_size = 64,
        hidden_sizes = [256, 512, 1024, 2048],
        depths = [3, 4, 6, 3],
        layer_type = "bottleneck",
        hidden_act = "relu",
        out_features = ["stage1"],
        num_labels = 5,
        #num_hidden_layers = 3,
    )

    efficientNet_custom_config = EfficientNetConfig(
        embedding_size = 64,
        hidden_sizes = [256, 512, 1024, 2048],
        width_coefficient = 2.0,
        depth_coefficient = 3.1,
        depths = [3, 4, 6, 3],
        layer_type = "bottleneck",
        hidden_act = "relu",
        out_features = ["stage1"],
        num_labels = 5,
        #num_hidden_layers = 3,
    )

    resnet_from_scratch = ResNetForImageClassification(resnet_custom_config)
    resnet_pretrained = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels = 5, ignore_mismatched_sizes=True)
    efficientnet_from_scratch = EfficientNetForImageClassification(efficientNet_custom_config)
    efficientnet_pretrained = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0", num_labels = 5, ignore_mismatched_sizes=True)
