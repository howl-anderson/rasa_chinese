###  addons_intent_classifier_textcnn_tf

基于 TextCNN (TensorFlow 版本) 的意图分类组件。
    
### addons_intent_classifier_textcnn_paddle

基于 TextCNN (PaddlePaddle 版本) 的意图分类组件。 
    
### addons_ner_bilstm_crf_tf

基于 Embedding+BiLSTM+CRF（TensorFlow 版本）的命名实体组件。
    
###  addons_ner_bilstm_crf_paddle

基于 Embedding+BiLSTM+CRF（PaddlePaddle 版本）的命名实体组件。

### bert_text_featurizer
  
基于 BERT 的文本特征向量提取组件（可用于意图分类）。
    
### bert_char_featurizer
  
基于 BERT 的字/词特征向量提取组件（可用于NER）。

### LanguageModelTokenizer

基于 HuggingFace's transformers 的分词组件。

pipeline 使用：
```yaml
  - name: "rasa_chinese.nlu.tokenizers.lm_tokenizer.LanguageModelTokenizer"
    tokenizer_url: "http://127.0.0.1:8000/"
```
NOTE: 需要使用 [rasa_chinese_service](https://github.com/howl-anderson/rasa_chinese_service) 作为服务器，在安装 `rasa_chinese_service` 后 （如何安装见 [rasa_chinese_service](https://github.com/howl-anderson/rasa_chinese_service) ），使用
```bash
python -m rasa_chinese_service.nlu.tokenizers.lm_tokenizer bert-base-chinese
```
启动 tokenizer 服务器


## 正在开发中

###  MicroAddonsTokenizer
   
基于[MicroTokenizer](https://github.com/howl-anderson/MicroTokenizer)的中文分词组件。
    
###  StackedBilstmTensorFlowPolicy
   
基于 Stacked Bi-LSTM 的对话 policy (TensorFlow 版本）

###  StackedBilstmPaddlePolicy

基于 Stacked Bi-LSTM 的对话 policy (PaddlePaddle 版本）