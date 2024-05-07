import sys
sys.path.insert(1, '../classifierModels')
import os
#from ..concatModels.OwLResNet import YOLOResNetModel ## for some reason relative imports are really annoying in python.
from flask import Flask, flash, render_template, request, redirect, session
import time

from werkzeug.utils import secure_filename
import torch
from torch import nn
from torchvision import transforms
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
    AutoImageProcessor,
    pipeline,
    ViTImageProcessor,
    PretrainedConfig,
)
from transformers.utils import ModelOutput
from torchvision import datasets
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import ViTConfig
import numpy as np
from PIL import Image
from safetensors.torch import load_model, save_model, load_file
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd




class YOLOResNetModel(nn.Module):
  def __init__(self,
               yolo,
               resnet,
               tokenizer,
               device = 'cuda', use_dropout=False):
    super().__init__()

    if torch.cuda.is_available() and device != 'cpu':
       self.device = 'cuda'
    else:
       self.device = 'cpu'

    self.yolo = yolo
    self.yolo.eval()
    #self.vit.to(device)
    
    self.resnet = resnet
    #self.resnet.to(device)
    self.resnet.classifier[1] = nn.Identity()
    self.resnet.eval()

    self.tokenizer = tokenizer

    self.dropout = nn.Dropout(p=use_dropout)
    
    self.concatenatedLayerSize = yolo.config.hidden_size + 2048
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


if __name__ == "__main__":
    
    ## loading model
    MODEL_FOLDER = "./best_models/yoloresnet_64_0.0003_False/checkpoint-246/"
    #MODEL_FOLDER = "./best_models/ViTForImageClassification_64_0.001_False/checkpoint-70/"

    resnet = ResNetForImageClassification.from_pretrained("./best_models/resnet_from_scratch_64_0.0003_False/checkpoint-346/")

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0,
        bias="none",
        modules_to_save=["classifier"],
    )
    """
    vit = ViTForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224",
                    num_labels=5,
                    ignore_mismatched_sizes=True,
                    output_attentions = True
            )
    """
    print("loaded pretrained model")


    #model = PeftModel.from_pretrained(vit, "./best_models/ViTForImageClassification_64_0.001_False/checkpoint-70/", is_trainable = False)

    yolo = YolosModel.from_pretrained("hustvl/yolos-small")
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
    yolo = get_peft_model(yolo, config)

    model = YOLOResNetModel(yolo = yolo, resnet = resnet, tokenizer = processor, device='cpu', use_dropout=0)
    load_model(model, MODEL_FOLDER + "model.safetensors")


    # getting dataset
    data_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])
    eval_data_dir = "../../data/dataset/val/"
    image_dataset = datasets.ImageFolder(eval_data_dir, data_transform)

    eval_dataloader = DataLoader(dataset = image_dataset, batch_size = len(image_dataset), shuffle = False)
    images, labels = next(iter(eval_dataloader))


    y_pred = []
    y_true = []
    model.eval()
    for inputs, labels in eval_dataloader:
            outputs = model(inputs)
            if isinstance(outputs, ModelOutput):
                outputs = outputs.logits
            predictions = torch.argmax(outputs, dim = 1)
            print(predictions)
            y_pred.extend(predictions)
            y_true.extend(labels)

    # constant for classes
    classes = ('1', '2', '3', '4', '5')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('output.png')