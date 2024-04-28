
# we want to load in a pretrained resnet model.
# we want to use the ImageFolder format specified by PyTorch
# we freeze the resnet parameters and train on our new dataset.
# train and evaluate
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
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
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    ResNetForImageClassification,
    Owlv2VisionModel,
    ResNetModel,
    AutoProcessor
)
import evaluate
import accelerate


cudnn.benchmark = True
plt.ion()   # interactive mode

## OK
## We need to just make this take in the pooled output of a OwLViT and a YOLO model, along with a resnet.
## Early fusion model.
#device = "cpu"  
class OwlResNetModel(nn.Module):
  def __init__(self,
               vit,
               resnet,
               tokenizer,
               device = 'cpu', use_dropout=False):
    super().__init__()

    self.device = device

    if torch.cuda.is_available():
       self.device = 'cuda'

    self.vit = vit
    self.vit.eval()
    #self.vit.to(device)
    
    self.resnet = resnet
    #self.resnet.to(device)
    self.resnet.eval()

    self.tokenizer = tokenizer

    self.dropout = nn.Dropout(p=use_dropout)
    
    self.concatenatedLayerSize = vit.config.hidden_size + 1000
    self.classifier = nn.Linear(self.concatenatedLayerSize, 5)

  def forward(self, pixel_values, labels = None):
      
      # Computing image embeddings
      image_embeddings = self.dropout(self.resnet(pixel_values).logits)
      #print("image embeddings shape: ", image_embeddings.shape)
      
      # Computing caption embeddings
      # tokenize all captions
      inputs = self.tokenizer(images = pixel_values, return_tensors="pt", do_rescale=False)

      #Pass the tokenized captions through the OwlViT model
      vit_output = self.dropout(self.vit(**inputs))

      #get the pooler output from the vit model's output
      pooled_output = vit_output.pooler_output

      # Concatenate image and caption embeddings along the batch dimension
      full_embeddings = torch.cat((image_embeddings, pooled_output), dim=1)

      logits = self.classifier(full_embeddings)
     
      if labels is not None:
        criterion = torch.nn.CrossEntropyLoss()#weight=torch.tensor([1.0, 2.0])
        loss = criterion(logits, labels)
        return (loss, logits)
      
      print(full_embeddings.shape)
      return (logits)
      

class YOLOResNetModel(nn.Module):
  def __init__(self,
               yolo,
               resnet,
               tokenizer,
               device = 'cpu', use_dropout=False):
    super().__init__()

    if torch.cuda.is_available():
       self.device = 'cuda'
    else:
       self.device = 'cpu'

    self.yolo = yolo
    self.yolo.eval()
    #self.vit.to(device)
    
    self.resnet = resnet
    #self.resnet.to(device)
    self.resnet.eval()

    self.tokenizer = tokenizer

    self.dropout = nn.Dropout(p=use_dropout)
    
    self.concatenatedLayerSize = yolo.config.hidden_size + 1000
    self.classifier = nn.Linear(self.concatenatedLayerSize, 5)

  def forward(self, pixel_values, labels = None):
      
      # Computing image embeddings
      image_embeddings = self.dropout(self.resnet(pixel_values).logits)
      
      # Computing caption embeddings
      # tokenize all captions
      inputs = self.tokenizer(images = pixel_values, return_tensors="pt", do_rescale=False).to(self.device)

      #Pass the tokenized captions through the OwlViT model
      yolo_output = self.yolo(**inputs)

      #get the pooler output from the vit model's output
      pooled_output = self.dropout(yolo_output.pooler_output)

      # Concatenate image and caption embeddings along the batch dimension
      full_embeddings = torch.cat((image_embeddings, pooled_output), dim=1)

      logits = self.classifier(full_embeddings)
     
      if labels is not None:
        criterion = torch.nn.CrossEntropyLoss()#weight=torch.tensor([1.0, 2.0])
        loss = criterion(logits, labels)
        return (loss, logits)
      
      return (logits)