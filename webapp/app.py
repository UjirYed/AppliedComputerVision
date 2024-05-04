import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../classifierModels')
from concatModels.OwLResNet import YOLOResNetModel
from flask import Flask, flash, render_template, request, redirect, session
import time
import os
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
    pipeline
)
from PIL import Image
from safetensors.torch import load_model, save_model
from peft import LoraConfig, get_peft_model

UPLOAD_FOLDER = os.path.abspath('uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BEST_MODEL = "yoloresnet_64_0.0003_False/checkpoint-246/"
MODEL_FOLDER = './saved_models/' + BEST_MODEL

data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # comment out for transformers
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # comment out for transformers
        ]),
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/occupancy", methods=['POST'])
def occupancy():
    if request.method == 'POST':
        f = request.files['file']

        if f.filename == '':
            flash('No selected file')
            return "No selected file"
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            # print(filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(image_path)
            prediction = evaluate_image(image_path)
            return str(prediction)

    print(request)
    return 'fail'

def evaluate_image(image_path):
    #model = pipeline("image-classification", model=MODEL_FOLDER)
    #model = ResNetForImageClassification.from_pretrained(MODEL_FOLDER, local_files_only=True)

    # Instantiate the classification model
    resnet = ResNetForImageClassification.from_pretrained("./saved_models/resnet_from_scratch_64_0.0003_False/checkpoint-346/")

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0,
        bias="none",
        modules_to_save=["classifier"],
    )

    yolo = YolosModel.from_pretrained("hustvl/yolos-small")
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
    yolo = get_peft_model(yolo, config)

    model = YOLOResNetModel(yolo = yolo, resnet = resnet, tokenizer = processor, device='cpu', use_dropout=0)
    load_model(model, MODEL_FOLDER + 'model.safetensors')

    image = data_transforms['val'](Image.open(image_path))
    input = {'pixel_values': image.unsqueeze(0)}
    outputs = model(**input)
    prediction = (torch.argmax(outputs))
    print(prediction)
    return int(prediction) + 1

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS