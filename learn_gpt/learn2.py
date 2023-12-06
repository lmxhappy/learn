# coding: utf-8
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='openai-gpt')
set_seed(42)
t = generator("The man worked as a", max_length=10, num_return_sequences=5)
print(t)


t = generator("The woman worked as a", max_length=10, num_return_sequences=5)
print(t)