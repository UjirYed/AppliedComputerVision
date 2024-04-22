import shap
import torch
from PIL import Image
from torchvision import datasets
import os
import torchvision.transforms as transforms
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)

image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=5,
    ignore_mismatched_sizes=True,
    output_attentions = True
)

def f(x):
    tmp = x.copy()
    tmp = torch.from_numpy(tmp)
    tmp = tmp.permute(0, 3, 1, 2)
    return vit(tmp).logits

if __name__ == "__main__":
    ##SHAP

# python function to get model output; replace this function with your own model function.

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
    class_names = image_datasets['train'].classes


    example_image = image_datasets['train'][4][0]

    example_tensor = image_processor(images = example_image, return_tensors = 'pt', do_rescale=False)['pixel_values']

    normalized_example_image = (example_tensor + 1) / 2
    example_image = normalized_example_image.permute(0, 3, 2, 1).numpy()
    print("example image type", type(example_image), example_image.shape, len(example_image))

    # define a masker that is used to mask out partitions of the input image.
    masker = shap.maskers.Image("inpaint_telea", (224,224,3))

    # create an explainer with model and image masker
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(
        example_image, max_evals=100, batch_size=50, outputs=shap.Explanation.argsort.flip[:4]
    )

    shap.image_plot(shap_values)
