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
    ResNetConfig,
    Owlv2VisionModel,
    AutoProcessor,
    ViTForImageClassification,
    Swinv2ForImageClassification,
    set_seed,
    YolosModel,
    AutoImageProcessor
)
from peft import LoraConfig, get_peft_model
from concatModels.OwLResNet import OwlResNetModel, YOLOResNetModel
import evaluate
import accelerate
import tqdm

set_seed(420)
models = ["yoloresnet"]

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
        p.requires_grad = False

def yes_grad(model):
    for p in model.parameters():
        p.requires_grad = True

def train(model, save_dir, batch_size, num_epochs, collate_fn, compute_metrics, trainOnlyHead, wandBProject, base_lr = 1e-3):
    os.environ["WANDB_PROJECT"] = f"<{wandBProject}>"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    training_args = TrainingArguments(
        output_dir = save_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",
        #logging_steps=10,
        save_total_limit=2,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        gradient_accumulation_steps=8,
        label_names=["labels"]
    )

    ## setting base learning rate
    base_lr = base_lr
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    training_args.learning_rate = base_lr * total_train_batch_size / 256

    print(torch.cuda.is_available())
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=image_datasets['train'],
        eval_dataset=image_datasets['val'],
        #tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )
    train_results = trainer.train()

    return train_results

def instantiate_model(model_name: str, use_dropout=False):
    if model_name not in models:
        raise Exception("Critical error: Model not valid. Please check your code.")
        return None

    if model_name == "ViTForImageClassification":
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=use_dropout,
            bias="none",
            modules_to_save=["classifier"],
        )
        vit = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=5,
                ignore_mismatched_sizes=True,
                output_attentions = True
        )
        model = get_peft_model(vit, config)
    
    if model_name == "Swinv2ForImageClassification":

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=use_dropout,
            bias="none",
            modules_to_save=["classifier"],
        )

        swin = Swinv2ForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256",
            num_labels = 5,
            ignore_mismatched_sizes = True
        )
        model = get_peft_model(swin, config)
        
    if model_name == "resnet_from_scratch":
        # Create custom resnet
        resnet_custom_config = ResNetConfig(
            embedding_size = 64,
            hidden_sizes = [256, 512, 1024, 2048],
            depths = [3, 4, 6, 3],
            layer_type = "bottleneck",
            hidden_act = "relu",
            out_features = ["stage1"],
            num_labels = 5,
            dropout_rate = use_dropout,
            #num_hidden_layers = 3,
        )
        model = ResNetForImageClassification(resnet_custom_config)
    
    if model_name == "resnet_pretrained":
        # Create pretrained resnet
        print("hello")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels=5, ignore_mismatched_sizes=True)
        no_grad(model)
        yes_grad(model.classifier)
    
    if model_name == "efficientnet_from_scratch":
        # Custom efficientnet
        efficientnet_custom_config = EfficientNetConfig(
            embedding_size = 64,
            hidden_sizes = [256, 512, 1024, 2048],
            width_coefficient = 2.0,
            depth_coefficient = 3.1,
            depths = [3, 4, 6, 3],
            layer_type = "bottleneck",
            hidden_act = "relu",
            out_features = ["stage1"],
            num_labels = 5,
            dropout_rate = use_dropout,
            #num_hidden_layers = 3,
        )
        model = EfficientNetForImageClassification(efficientnet_custom_config)

    if model_name == "efficientnet_pretrained":
        # Pretrained efficientnet
        model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0", num_labels=5, ignore_mismatched_sizes=True)
        no_grad(model)
        yes_grad(model.classifier)

    if model_name == "owlresnet":
        ## Instantiate the Owl/resnet concatenation model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        #vit = Owlv2VisionModel.from_pretrained("google/owlv2-base-patch16")
        vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")#.to(device)
        model = OwlResNetModel(vit = vit, resnet = resnet, tokenizer = processor, use_dropout=use_dropout)
        no_grad(model)
        yes_grad(model.classifier)

    if model_name == "yoloresnet":
        ## Instantiate the yolo/resnet concatenation model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        yolo = YolosModel.from_pretrained("hustvl/yolos-small")
        processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
        model = YOLOResNetModel(yolo = yolo, resnet = resnet, tokenizer = processor, use_dropout=use_dropout)
        no_grad(model)
        yes_grad(model.classifier)
        
    return model

def ImageFolderDataSets(data_dir, data_transforms):
    """
    returns dictionary of train and val datasets
    """
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    return image_datasets


if __name__ == "__main__":
    # Create the data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # comment out for transformers
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # comment out for transformers
        ]),
    }

    # loading training and validation datasets
    image_datasets = ImageFolderDataSets("../data/dataset/", data_transforms)

    ''' owlresnet_train_dict = {
    "model": owlresnet,
    "save_dir": "pretrained_resnet",
    "batch_size": 4,
    "num_epochs":  50,
    "collate_fn": collate_fn,
    "compute_metrics": compute_metrics,
    "trainOnlyHead": True,
    "wandBProject": "XXX",
    }'''

    base_rates = [1e-3, 3e-4]
    batch_sizes = [64, 128]
    dropouts = [False, .2]

    for modelName in models:
        for size in batch_sizes:
            for base_lr in base_rates:
                for dropout in dropouts:
                    trial_name = modelName + "_" + str(size) + "_" + str(base_lr) + "_" + str(dropout)

                    modelDict = {
                    "model": instantiate_model(modelName, use_dropout=dropout),
                    "save_dir": "saved_models/" + trial_name,
                    "batch_size": size,
                    "num_epochs":  100,
                    "collate_fn": collate_fn,
                    "compute_metrics": compute_metrics,
                    "trainOnlyHead": "ambiguous",
                    "wandBProject": trial_name,
                    "base_lr": base_lr
                    }

                    print("Beginning training - " + trial_name + ": \n")

                    train(**modelDict)