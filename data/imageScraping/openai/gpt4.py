import openai

api_key = "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"

client = openai.Client(api_key=api_key)

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a helpful assistant that creates song lyrics."},
    {"role": "user", "content": "Generate two verses and a chorus for the following scenario: this is a taylor swift song about looking for love in the college dining hall, food is disgusting, columbia university"}
  ]
)

print(completion.choices[0].message)


