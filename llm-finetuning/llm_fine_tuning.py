# -*- coding: utf-8 -*-
"""llm_fine_tuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BQbfHbnJNEAI7VOBCyPpeR9ZzFR4aC9J
"""

!pip install openai

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-capture-your-key",
)

client.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="fine-tune"
)

client.fine_tuning.jobs.create(
  training_file="file-JRuNblQF3Dojn16NlPGC7dhX",
  model="gpt-3.5-turbo"
)

# Retrieve the state of a fine-tune
#ftjob-QcFEtWj3NwCuVuqFCVrysVXt
client.fine_tuning.jobs.retrieve("ftjob-Oi0owJPFg3PIiT4Ptlcrvm8C")

"""Original GPT 3.5 Turbo model"""

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
      {"role": "assistant", "content": "You know everrything"},
    {"role": "user", "content": "what are sustainability initiatives?"}
  ]
)
print(completion.choices[0].message)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "assistant", "content": "You know everrything"},
    {"role": "user", "content": "Can you provide details about XYZ Company's latest sustainability initiatives and how they align with the company's long-term goals?"}
  ]
)
print(completion.choices[0].message)

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::97VdN55q",
  messages=[
    {"role": "assistant", "content": "Marv is a knowledgeable spokesperson or representative of XYZ Company."},
    {"role": "user", "content": "what are sustainability initiatives?"}
  ]
)
print(completion.choices[0].message)

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::97VdN55q",
  messages=[
    {"role": "assistant", "content": "Marv is a knowledgeable spokesperson or representative of XYZ Company."},
    {"role": "user", "content": "what do you know about xyz company?"}
  ]
)
print(completion.choices[0].message)

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::97VdN55q",
  messages=[
    {"role": "assistant", "content": "Marv is a knowledgeable spokesperson or representative of XYZ Company."},
    {"role": "user", "content": "what are some of the sustainbilityy initiatives by the xyz company?"}
  ]
)
print(completion.choices[0].message)

