import torch
from torch import nn

## OK
## We need to just make this take in the pooled output of a OwLViT and a YOLO model, along with a resnet.
## Early fusion model.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cpu"  
class BaseLineModel(nn.Module):
  def __init__(self,
               resnetModel,
               owlvit,
               tokenizer,
               device):
    super().__init__()


    self.resnet = resnetModel
    self.resnet.to(device)
    self.resnet.eval()
    self.bertModel = bertModel
    self.bertModel.eval()
    self.bertModel.to(device)
    self.tokenizer = tokenizer
    
    self.concatenatedLayerSize = bert_config.hidden_size + 512
    self.clf = nn.Linear(self.concatenatedLayerSize, 2)
    #print(self.concatenatedLayerSize)

  def forward(self, x):
      images, captions = x
      # Computing image embeddings
      images = images.to(device)
      image_embeddings = self.resnet(images)
         
      # Computing caption embeddings
      # tokenize all captions
      tokenized_captions = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
      tokenized_captions = tokenized_captions.to(device)
      #Pass the tokenized captions through the BERT model
      bert_output = self.bertModel(**tokenized_captions)

      #get the pooler output from the BERT model's output
      pooled_output = bert_output.pooler_output

      # Concatenate image and caption embeddings along the batch dimension
      full_embeddings = torch.cat((image_embeddings, pooled_output), dim=1)
      return self.clf(full_embeddings)