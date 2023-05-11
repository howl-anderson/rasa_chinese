# rasa_chinese

rasa_chinese 是专门针对中文语言的 [rasa](https://github.com/RasaHQ/rasa) 组件扩展包。提供了一些针对中文语言的组件。

**本软件包得到了 Rasa 官方的认可，官方博客中推荐中文 Rasa 用户使用： <https://rasa.com/blog/non-english-tools-for-rasa/>**

## 安装

```bash
pip install rasa_chinese
```

## 当前包含的组件

### LanguageModelTokenizer

基于 HuggingFace's transformers 的分词组件。

pipeline 使用：

```yaml
pipeline:
  - name: "rasa_chinese.nlu.tokenizers.lm_tokenizer.LanguageModelTokenizer"
```

LanguageModelTokenizer 的分词方法必须和 LanguageModelFeaturizer 保持一致。

如果用户在 pipeline 中指定了 LanguageModelFeaturizer 的参数，那么也需要为 LanguageModelFeaturizer 设置相同的参数。如下所示:

```yaml
pipeline:
  - name: "rasa_chinese.nlu.tokenizers.lm_tokenizer.LanguageModelTokenizer"
    # 以下的参数必须和 LanguageModelFeaturizer 的参数保持完全一致
    model_name: "roberta"
    model_weights: "roberta-base"
  - name: LanguageModelFeaturizer
    model_name: "roberta"
    model_weights: "roberta-base"
```
