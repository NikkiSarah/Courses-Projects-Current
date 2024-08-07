# -*- coding: utf-8 -*-
"""02 Open-Source LLMs.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TSLOmC3E-P3BkwvYxDIZFzQW_Mq8yfDb

## **02 Open-Source LLMs**
"""

# install libraries
!pip install -U transformers accelerate bitsandbytes sentencepiece

# view the specs of the GPU
!nvidia-smi

# get the minsearch python file
!wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py

# load, parse and index the data
import requests
import minsearch

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)

# define functions for retrieval and RAG
def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results


def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

# check how much space is in the home directory
!df -h

# OPTIONAL: tell Google Cloud to use a different directory (need to restart kernel)
import os
os.environ['HF_HOME'] = '/opt/bin/.nvidia/cache/'

"""### Google FLAN T5"""

# run the FLAN-T5-XL model example code
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokeniser = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

# view the input ids
input_text = "translate English to German: How old are you?"
input_ids = tokeniser(input_text, return_tensors="pt").input_ids.to("cuda")
print(input_ids)

# view the output
outputs = model.generate(input_ids)
print(tokeniser.decode(outputs[0]))

# put it all together
input_text = "translate English to German: How old are you?"

def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT: {context}
    """.strip()

    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    input_ids = tokeniser(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    results = tokeniser.decode(outputs[0])
    return results

rag("I just discovered the course. Can I still join it?")

# have another go at the "G" part
def llm(prompt, generate_params=None):
    if generate_params is None:
        generate_params = {}

    input_ids = tokeniser(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        max_length=generate_params.get("max_length", 100),
        num_beams=generate_params.get("num_beams", 5),
        do_sample=generate_params.get("do_sample", False),
        temperature=generate_params.get("temperature", 1.0),
        top_k=generate_params.get("top_k", 50),
        top_p=generate_params.get("top_p", 0.95),
    )
    result = tokeniser.decode(outputs[0], skip_special_tokens=True)
    return result

rag("I just discovered the course. Can I still join it?")

"""### Microsoft Phi 3 Mini

**Note: Restart the session to free up GPU memory.**
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

# download the model and its tokeniser
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokeniser = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# combine them into a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokeniser,
)

messages = [
    {"role": "user", "content": "I've just discovered the course. Can I still join?"},
]

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])

# amend the prompt so the answers are more relevant/less generic
def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    return output[0]['generated_text'].strip()

rag("I just dicovered the course. Can I still join it?")

"""### Mistral-7B

**Note: Restart the session to free up GPU memory.**
"""

# authenticate to the HuggingFace API
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get('HF_TOKEN'))

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
)
tokeniser = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")

model_inputs = tokeniser(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs)
tokeniser.batch_decode(generated_ids, skip_special_tokens=True)[0]

# amend the prompt so the answers are more relevant/less generic
def build_prompt(query, search_results):
    prompt_template = """
    QUESTION: {question}

    CONTEXT:
    {context}

    ANSWER:
    """.strip()

    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    model_inputs = tokeniser([prompt], return_tensors="pt").to("cuda")
    generated_ids = model.generate(**model_inputs, max_length=700)
    result = tokeniser.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return result

rag("I just discovered the course. Can I still join it?")

# make some improvements
from transformers import pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokeniser)

def build_prompt(query, search_results):
    prompt_template = """
    QUESTION: {question}

    CONTEXT:
    {context}

    ANSWER:
    """.strip()

    context = ""

    for doc in search_results:
        context = context + f"question: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = generator(prompt, max_length=500, temperature=0.7, top_p=0.95,
                         num_return_sequences=1)
    response_final = response[0]['generated_text']
    return response_final[len(prompt):].strip()

rag("I just discovered the course. Can I still join it?")

