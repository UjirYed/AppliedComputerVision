from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    EfficientNetConfig,
    EfficientNetForImageClassification,
    ViTForImageClassification,
    AutoTokenizer,
    EfficientNetImageProcessor,
    ViTImageProcessor,
)
from PIL import Image

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
import os


## This code defines functions that compue and plot attention maps based on the attention matrices found in transformer models.
## Inspired by https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb 
def compute_attention_map(modelOutputs): #make sure to pass in a output that has an attentions attribute!!
    attention_matrices = torch.stack([modelOutputs.attentions[i] for i in range(len(modelOutputs.attentions))])
    attention_matrices = attention_matrices.squeeze(1)

    att_mat = attention_matrices
    att_mat = torch.mean(att_mat, dim=1) #taking average of attention across all heads
    #print("shape of att_mat after averaging", att_mat.shape)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    #print("residual att size", att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    #print("aug_att_mat shape", aug_att_mat.shape)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    #print("joint attentions shape:", joint_attentions.shape)

    #print("shape of augmented attention matrix:", aug_att_mat.size(0))
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions[0])
    # Attention from the output token to the input space.
    v = joint_attentions[-1]

    #print("values of v", v)
    #print("shape of v", v.shape)
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    #print("grid size", grid_size)
    attention_map = mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), (224,224))[..., np.newaxis]
    mask = np.repeat(mask, 3, axis=-1)  # Replicate mask across channels
    mask = np.transpose(mask, (2, 0, 1))  # Transpose mask to match the order of dimensions in test_image

    return mask

def plot_attention_map(input_image, mask): #pass in the image that you used as input to the model.
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 8))
    #print(mask[0].shape)
    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    ax1.imshow(input_image.squeeze(0).permute(1, 2, 0))
    ax2.imshow(mask[0], cmap="viridis")
    plt.show()


if __name__ == "__main__":


    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=5,
        ignore_mismatched_sizes=True,
        output_attentions = True
    )

    # For straightforward datasets, sometimes you can make do with built-in PyTorch dataset objects.
    # We want to apply automated data augmentations, which will be different for the training
    # and eval scenarios

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
    }
    data_dir = "../../data/dataset/"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    test_image = image_datasets['train'][4][0]

    inputs = image_processor(images = test_image, return_tensors = 'pt', do_rescale = False)
    # we will use this test image to do all our preliminary testing to make sure stuff works.
    test_image = test_image.unsqueeze(0)

    mask = compute_attention_map(vit(test_image))

    plot_attention_map(test_image, mask)



