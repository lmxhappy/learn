# coding: utf-8
from sentence_transformers import SentenceTransformer, models
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def my_test():


    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

    tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    output = model(**tokens)

    print(output)

    
def my_test2():
    # st = SentenceTransformer('lmxhappy/yule_bagua_bert')
    tokenizer = AutoTokenizer.from_pretrained('lmxhappy/yule_bagua_bert')

    text = '国内旅行常见风险分析】1.意外受伤：交通意外、意外磕碰、意外骨折、意外擦伤等；突发急性病：急性肠胃炎、突发心血管疾病等；3.紧急救援：遭受意外伤害或罹患疾病，紧急香港概念第一币运送和送返；4.高风险运动意外：拓展活动、场地趣味活动、露营、游泳等。#保险##意外##健康##重疾险##香港概念第一币##养老#'
    # ret = st.tokenize([text])
    # print(st.tokenizer)
    # print(ret)
    
    ret = tokenizer(text)
    print(ret)


if __name__ == '__main__':
    my_test2()