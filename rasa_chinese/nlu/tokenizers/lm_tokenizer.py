import logging
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class LanguageModelTokenizer(Tokenizer):
    """HuaggingFace's Transformers based tokenizer."""

    defaults = {
        # URL to tokenizer service
        "tokenizer_url": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:  # noqa: D107
        import requests

        super().__init__(component_config)

        self.session = requests.Session()

    @classmethod
    def required_packages(cls) -> List[Text]:  # noqa: D102
        return ["requests"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:  # noqa: D102
        text = message.get(attribute)

        response = self.session.get(self.component_config["tokenizer_url"], params={"q": text})
        tokenization_result = response.json()

        tokens = []

        for token_text, start, end in tokenization_result:
            token = Token(token_text, start, end)
            tokens.append(token)

        return self._apply_token_pattern(tokens)
