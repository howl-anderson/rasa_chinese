import logging
import typing
import copy
from typing import Any, Dict, List, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    pass


class MicroAddonsTokenizer(Tokenizer):
    provides = ["tokens"]

    language_list = ["zh"]

    defaults = {
        "custom_dict": None,
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
    }  # default don't load custom dictionary

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super().__init__(component_config)

        kwargs = copy.deepcopy(component_config)
        kwargs.pop("name")

        self.custom_dict = kwargs.pop("custom_dict", None)

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
