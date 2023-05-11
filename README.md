# rasa_chinese

rasa_chinese 是专门针对中文语言的 [rasa](https://github.com/RasaHQ/rasa) 组件扩展包。提供了许多针对中文语言的组件。

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
  - name: "rasa_chinese.nlu.tokenizers.lm_tokenizer.LanguageModelTokenizer"
    tokenizer_url: "http://127.0.0.1:8000/"
```

NOTE: 需要使用 [rasa_chinese_service](https://github.com/howl-anderson/rasa_chinese_service) 作为服务器，在安装 `rasa_chinese_service` 后 （如何安装见 [rasa_chinese_service](https://github.com/howl-anderson/rasa_chinese_service) ），使用

```bash
python -m rasa_chinese_service.nlu.tokenizers.lm_tokenizer bert-base-chinese
```

启动 tokenizer 服务器

### [已废弃] MicroTokenizer

基于 MicroTokenizer (https://github.com/howl-anderson/MicroTokenizer) 的分词组件

pipeline 使用：

```yaml
  - name: "rasa_chinese.nlu.tokenizers.MicroTokenizer"
```

### [已废弃] 微信 connector

当前(未来可能会改变),我们可以直接使用 rasa 自带的 rest channel connector 来完成和 Rasa adapter 的连接. 因此只需确保 rast channel (位于`credentials.yml`文件中) 是开启的.
当前微信 connector 配置的核心位于 [rasa_chinese_service](https://github.com/howl-anderson/rasa_chinese_service) 仓库, 用户可以仔细阅读相关文档,按照文档逐步设置. 

### [已废弃] 离线模型

基于 transformers 的组件需要下载模型。这一过程需要访问 AWS 服务器，对于国内的用户来说可能速度慢或者网络不稳定。这里提供了工具可以直接下载模型做成离线模型。离线模型可以直接在组件中使用。

#### 离线模型下载

```bash
python -m rasa_chinese.tools.download_transformers_model bert-base-cased offline_model
```

其中 `bert-base-cased` 是你要下载的模型名字， `offline_model` 是你离线模型存储的目录，目录必须已经存在。

注意：你需要在网络通畅（指能顺利访问AWS服务）的主机上运行该命令，否则可能会出现网络错误。

#### 离线模型使用

在你的 config.yml 文件中，这样使用离线模型

```yaml
pipeline:
  - name: JiebaTokenizer
  - name: LanguageModelFeaturizer
    model_name: bert
    model_weights: /path/to/offline_model
  - name: "DIETClassifier"
    epochs: 100
```

其中的 `/path/to/offline_model` 指向你的模型目录

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
