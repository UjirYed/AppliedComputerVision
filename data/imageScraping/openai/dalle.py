import openai

api_key = "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"

openai.api_key = api_key


client = openai.Client(api_key=api_key)

response = client.images.generate(
  model="dall-e-3",
  prompt="college dining hall with a few empty tables, but some tables are full.",
  size="1024x1024",
  quality="standard",
  style="natural",
  n=1
)

image_url = response.data[0].url
print(image_url)