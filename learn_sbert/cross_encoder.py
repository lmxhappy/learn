# coding: utf-8
from sentence_transformers import CrossEncoder
model = CrossEncoder('lmxhappy/yule_bagua_bert', max_length=512)
scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])
print(scores)