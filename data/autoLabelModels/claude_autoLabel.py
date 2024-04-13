# we want to read line by line from each file
# then we want to load it in if it is a link. using a http request and then encoding using base64
# then, we call claude on the image that we have just retrieved.
# we extract the label from the response, and then store it in the appropriate place within the dataset folder
#

import anthropic
import base64
import httpx
import typing
import time

#Creating our claude client.
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)


def load_image(image_link: str) -> str:
    """
    Takes in image data and returns an encoding of the image in base64.
    """
    # did we retrieve a link or the actual image data when scraping google?
    if "https" in image_link:
        image_data = base64.b64encode(httpx.get(image_link).content).decode("utf-8")
    else:
        image_data = image_link.split(',')
        image_data = image_data[1]
    return image_data

def autolabel(image_link_file_path, prompt, label_file_path):
    with open(image_link_file_path) as f:
        image_links =  f.read().splitlines()
    label_file = open(label_file_path, "a")
    for image_link in image_links:
        image_data = load_image(image_link)

        print(image_data)
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
            label = message.content[0].text[-1] + "\n"
        except Exception as e:
            print(f"Error writing label for {image_link}: {str(e)}")
            label = "N/A\n"
        time.sleep(10)
        label_file.write(label)

if __name__ == "__main__":
    image_link_file_path = "image_link_test.txt"
    label_file_path = "label_file.txt"
    prompt = "USER: Classify on a scale of 1-5 how busy this dining hall looks based on both the number of people and empty spaces (like chairs and tables). If there are no people and lots of empty available areas for them to be in, then it is empty. Classify as: 1-sparse/empty, 2-not too busy, 3-moderately busy, 4-busy, 5-extremely busy. Return just 1 number. After classifying try one more time, thinking deeply, not getting distracted by irrelevant details of the dining hall like architecture, only the busyness. At the end, print your final number at the end of your response.:"
    with open(image_link_file_path) as f:
        image_links = f.read().splitlines()
    

    autolabel(image_link_file_path, prompt, label_file_path)