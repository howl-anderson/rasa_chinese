import copy
import logging
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class MicroTokenizer(Tokenizer):
    provides = ["tokens"]

    language_list = ["zh"]

    defaults = {
        "dictionary_path": None,  # default don't load custom dictionary
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super().__init__(component_config)

        kwargs = copy.deepcopy(component_config) if component_config else dict()

        self.custom_dict = kwargs.pop("dictionary_path", None)

        if self.custom_dict:
            self.load_custom_dictionary(self.custom_dict)

    @staticmethod
    def load_custom_dictionary(custom_dict: Text) -> None:
        import MicroTokenizer

        MicroTokenizer.load_userdict(custom_dict)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["MicroTokenizer"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        import MicroTokenizer

        text = message.get(attribute)

        tokenized = MicroTokenizer.cut(text)

        tokens = []
        offset = 0
        for word in tokenized:
            tokens.append(Token(word, offset))
            offset += len(word)

        return tokens
