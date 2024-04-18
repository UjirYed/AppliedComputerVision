import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from transformers import AutoProcessor, Owlv2VisionModel, AutoImageProcessor, YolosModel
from torchinfo import summary

device = "cuda:0" if torch.cuda.is_available() else "cpu"

owlvit = Owlv2VisionModel.from_pretrained("google/owlv2-base-patch16")
owl_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")

print(summary(owlvit))

print(owlvit.config.hidden_size)

yolovit = YolosModel.from_pretrained("hustvl/yolos-small")
yolo_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")

class ConcatenationModel(nn.Module):
    def __init__(self, resnet, object_detector):
        super().__init__()

        # Save the resnet and object detector to the model.
        self.resnet = resnet
        self.object_detector = object_detector

        # Add the hidden sizes of the resnet and object detector 
        # together to determine the size of the classifier layer.
        self.classifier_size = object_detector.config.hidden_size + resnet.config.hidden_size #resnet.config may not be the play here i forget exactly how this worked...
        self.clf = nn.Linear(self.classifier_size, 5)
    
    ###uhhhhh not sure if this works but it is a start
    def forward(self, pixel_values):
        # Get the resnet embedding
        resnetEmbed = self.resnet(pixel_values)
        # Get the object detector embedding
        objectEmbed = self.object_detector(pixel_values)
        # Concatenate the embeddings
        embeddings = torch.cat((resnetEmbed, objectEmbed), 1)
        # Pass the embeddings through the classifier layer.
        return self.clf(embeddings)