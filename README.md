# rasa_chinese

rasa_chinese 是专门针对中文语言的 [rasa](https://github.com/RasaHQ/rasa) 组件扩展包。提供了许多针对中文语言的组件。

## 安装
```bash
pip install rasa_chinese
```

## 当前包含的组件
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

### MicroTokenizer

基于 MicroTokenizer (https://github.com/howl-anderson/MicroTokenizer) 的分词组件

pipeline 使用：
```yaml
  - name: "rasa_chinese.nlu.tokenizers.MicroTokenizer"
```

### 微信 connector
当前(未来可能会改变),我们可以直接使用 rasa 自带的 rest channel connector 来完成和 Rasa adapter 的连接. 因此只需确保 rast channel (位于`credentials.yml`文件中) 是开启的.
当前微信 connector 配置的核心位于 [rasa_chinese_service](https://github.com/howl-anderson/rasa_chinese_service) 仓库, 用户可以仔细阅读相关文档,按照文档逐步设置. 

## 正在开发中

更多组件正在从 1.x 版本移植到 2.x 版本。
    

## 如何使用
将组件的全路径类名放到 config.yaml 中.

例如下面这样：
```yaml
language: "zh"

pipeline:
  - name: "rasa_chinese.nlu.TensorflowNLP"
  - name: "rasa_chinese.nlu.BilstmCrfTensorFlowEntityExtractor"
    max_steps: 600
  - name: "rasa_chinese.nlu.TextCnnTensorFlowClassifier"
    max_steps: 600

policies:
  - name: MemoizationPolicy
  - name: rasa_chinese.core.StackedBilstmTensorFlowPolicy
```
