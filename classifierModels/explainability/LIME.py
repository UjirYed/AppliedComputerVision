import torch
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)

def model_predict(inputs):
    # Convert the inputs to a PyTorch tensor
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)  # add a batch dimension if it doesn't exist.
    print("inputs shape", inputs.shape)  # (B, H, W, C)
    inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float()

    # Get the model's output logits
    with torch.no_grad():
        logits = vit(inputs).logits

    # Convert the logits to probabilities
    probs = torch.softmax(logits, dim=1)

    # Return the probabilities as a NumPy array
    return probs.cpu().numpy()


if __name__ == "__main__":
    vit = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=5,
    ignore_mismatched_sizes=True
    )   
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # Load an example image
    test_image = Image.open("../../data/dataset/train/1/File 00008.jpg")
    inputs = image_processor(images = test_image, return_tensors = 'pt')
    example_tensor = inputs['pixel_values'].squeeze(0)

    # Get the model prediction for the example image
    model_output = vit(example_tensor.unsqueeze(0))
    print(model_output)
    predicted_class = torch.argmax(model_output.logits)

    #create lime explainer
    explainer = lime_image.LimeImageExplainer()

    #compute explanation
    explanation = explainer.explain_instance(np.transpose(example_tensor.numpy(), (1, 2, 0)), model_predict, top_labels=5, hide_color=0, num_samples=1000)

    #get explanation for the predicted class
    idx_predicted_class = list(explanation.top_labels).index(predicted_class.item())
    temp, mask = explanation.get_image_and_mask(idx_predicted_class, positive_only=True, num_features=5, hide_rest=False)

    #plot the explanation
    print("temp shape:", temp.shape)
    print("mask shape", mask.shape)
    mask_grayscale = np.mean(mask, axis = -1).astype(np.int32)


    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 8))

    ax1.imshow(mark_boundaries(temp, mask_grayscale))
    ax2.imshow(mask)
    plt.show()