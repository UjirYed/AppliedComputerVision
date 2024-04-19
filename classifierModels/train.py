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

def compute_metrics(p):
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def no_grad(model):
    for p in model.parameters():
        p.requires_grad= False

def yes_grad(model):
    for p in model.parameters():
        p.requires_grad= True

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

    # Instantiate each model into a dictionary with the model and its corresponding save directory
    resnet_from_scratch = {"model": ResNetForImageClassification(resnet_custom_config),
                           "save_dir": "./resnet_from_scratch"}

    resnet_pretrained = {"model": ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels = 5, ignore_mismatched_sizes=True),
                         "save_dir": "./resnet_pretrained"}
    no_grad(resnet_pretrained["model"])
    yes_grad(resnet_pretrained["model"].classifier)

    efficientnet_from_scratch = {"model": EfficientNetForImageClassification(efficientNet_custom_config),
                                "save_dir": "./efficientnet_from_scratch"}
    
    efficientnet_pretrained = {"model": EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0", num_labels = 5, ignore_mismatched_sizes=True),
                               "save_dir": "./efficientnet_pretrained"}
    no_grad(efficientnet_pretrained["model"])
    yes_grad(efficientnet_pretrained["model"].classifier)

    # Create the training args
    training_args = TrainingArguments(
        output_dir = resnet_pretrained["save_dir"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=100,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        #gradient_accumulation_steps=8,
    )

    # Compute the learning rate from the base learning rate
    base_learning_rate = 1e-3
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )

    training_args.learning_rate = base_learning_rate * total_train_batch_size / 256

    # Create the trainer
    trainer = Trainer(
        model=resnet_pretrained["model"],
        args=training_args,
        train_dataset=image_datasets['train'],
        eval_dataset=image_datasets['val'],
        #tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    print(torch.cuda.is_available())

    train_results = trainer.train()

    print(train_results)