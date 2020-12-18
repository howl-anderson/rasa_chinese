from bert_serving.client import BertClient
bc = BertClient()
result = bc.encode(['First do it', 'then do it right', 'then do it better'])

print(result)

result = bc.encode(['你是谁', '我是王小明', '知识就是力量'], is_tokenized=False)

print(result)

result = bc.encode(['我是王小明'], is_tokenized=False)

print(result)