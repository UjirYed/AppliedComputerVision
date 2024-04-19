# we want to read line by line from each file
# then we want to load it in if it is a link. using a http request and then encoding using base64
# then, we call claude on the image that we have just retrieved.
# we extract the label from the response, and then store it in the appropriate place within the dataset folder
#
import anthropic
import base64
import typing
import time
import os
import shutil
import requests
import tqdm
import re
#Creating our claude client.
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)


def load_image(image_link: str, base_64 = False):
    """
    Takes in image path and loads in the image. If base_64 is specified, then a string encoding in base_64 is returned. 
    """
    # did we retrieve a link or the actual image data when scraping google?
    if "https" in image_link:
        image_data = requests.get(image_link).content
        if base_64:
            image_data = base64.b64encode(image_data).decode("utf-8")
    else:
        image_data = image_link.split(',')
        image_data = image_data[1]
    return image_data

def autolabel(image_link_file_path, prompt, label_file_path):
    with open(image_link_file_path) as f:
        image_links =  f.read().splitlines()
    
    """
    if os.path.exists(label_file_path): #clear file before writing to it.
        with open(label_file_path, "w"):
            pass
    """
    label_file = open(label_file_path, "w")
    for image_link in image_links:
        image_data = load_image(image_link, base_64=True)

        try:
            message = client.messages.create(
            model="claude-3-sonnet-20240229", #claude-3-haiku-20240307
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
            )
            text_response = message.content[0].text
            print(type(text_response))
            print(text_response)
            pattern = r'\$(\d+)\$'

            match = re.search(pattern, text_response)
            if match:
                label = match.group(1)
            else:
                label = "N/A"
            print(label)
        except Exception as e:
            print(f"Error writing label for {image_link}: {str(e)}")
            label = "error"
        time.sleep(10)
        label_file.write(label + "\n")

def move_images_to_dataset(image_link_file_path: str, label_file_path: str, dataset_path: str, acceptable_labels = [], offset=0) -> None:
    """
    Takes in image data file and corresponding label file. Loads actual images into a directory with the PyTorch ImageFolder structure.
    """
    with open(image_link_file_path) as f:
        image_links =  f.read().splitlines()

    with open(label_file_path) as f:
        labels = f.read().splitlines()
    
    for i, (image_link, label) in enumerate(zip(image_links, labels)):
        """
        if label not in acceptable_labels: # throw away classifications that were erroneously produced by the autoLabeller
            print("unnaceptable label of:" + label)
            continue
        """
        image = load_image(image_link)
        
        label_dir = os.path.join(dataset_path, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        image_filename = os.path.basename(f"{i+offset}.jpeg")
        image_path = os.path.join(label_dir, image_filename)

        if isinstance(image, str):
            image = base64.b64decode(image)

        with open(image_path, 'wb') as img_file:
            img_file.write(image)
            print(f"{i+offset}.jpeg saved to {label_dir}")



if __name__ == "__main__":
    print(os.path.exists("../imageScraping/LLMBrowser/image_urls/outputaa"))

    image_link_file_path = "../imageScraping/LLMBrowser/image_urls/outputaa"
    label_file_path = "label_file.txt"
    dataset_path = "testdataset"
    acceptable_labels = ['0','1','2','3','4','5', 'N/A']

    
    prompt = "Classify on a scale of 0-5 how busy this dining hall looks, based on both the number of people and empty spaces (like chairs and tables) and whether it actually is a dining hall at all. If there are no people and lots of empty available areas for them to be in, then it is empty. Classify as: 0 - not even remotely a dining hall or is just a picture of food without people or seating in it, 1-sparse/empty, 2-not too busy, 3-moderately busy, 4-busy, 5-extremely busy. Return just 1 number. After classifying, try one more time, not getting distracted by irrelevant details of the dining hall like architecture, only the busyness. Then, provide ONLY the final classification number in the format $classification$."
    with open(image_link_file_path) as f:
        image_links = f.read().splitlines()

    autolabel(image_link_file_path, prompt, label_file_path)
    move_images_to_dataset(image_link_file_path, label_file_path, dataset_path, acceptable_labels=acceptable_labels)