import sys


def main(model_name="bert-base-cased", output_dir="offline_model"):
    try:
        from transformers import TFBertModel, BertTokenizer
    except ImportError:
        print("使用本工具，必须要先安装 transfomers 库，你可以通过 pip install rasa[transformer] 安装")
        sys.exit(-1)

    model = TFBertModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    model_name = sys.argv[1]
    output_dir = sys.argv[2]
    main()
