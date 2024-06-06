# -*- coding: utf-8 -*-
"""openai_gpt-4o.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OTDfjpAQ2n423POfk8qNzavwN8RayOZR
"""

!pip install openai

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

def display_chat_history(messages):
    for message in messages:
        print(f"{message['role'].capitalize()}: {message['content']}")

def get_assistant_response(messages):
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    response = r.choices[0].message.content
    return response

messages = [{"role": "assistant", "content": "How can I help?"}]

import base64

image_path = "openai.png"

# We will encode the image to a base64 string
def encode_image(image):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(image_path)



response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe the key idea of the image"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ]
)
print(response.choices[0].message.content)

