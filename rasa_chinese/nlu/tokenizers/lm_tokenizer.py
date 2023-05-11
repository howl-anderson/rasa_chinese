import logging
from typing import Any, Dict, List, Optional, Text

import rasa.shared.utils.io
import rasa.utils.io

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class LanguageModelTokenizer(Tokenizer):
    """HuaggingFace's Transformers based tokenizer."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
            # following settings are same as LanguageModelFeaturizer
            "model_name": "bert",
            "model_weights": None,
            "cache_dir": None,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        super().__init__(config)
        self._load_model_metadata()
        self._load_model_instance()

    def _load_model_metadata(self) -> None:
        """
        COPY and simplified from `LanguageModelFeaturizer._load_model_metadata`
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,
            model_weights_defaults,
        )

        self.model_name = self._config["model_name"]

        if self.model_name not in model_class_dict:
            raise KeyError(
                f"'{self.model_name}' not a valid model name. Choose from "
                f"{str(list(model_class_dict.keys()))} or create"
                f"a new class inheriting from this class to support your model."
            )

        self.model_weights = self._config["model_weights"]
        self.cache_dir = self._config["cache_dir"]

        if not self.model_weights:
            logger.info(
                f"Model weights not specified. Will choose default model "
                f"weights: {model_weights_defaults[self.model_name]}"
            )
            self.model_weights = model_weights_defaults[self.model_name]

    def _load_model_instance(self) -> None:
        """
        COPY and simplified from `LanguageModelFeaturizer._load_model_instance`
        """
        from rasa.nlu.utils.hugging_face.registry import (
            model_class_dict,
            model_tokenizer_dict,
        )

        logger.debug(f"Loading Tokenizer and Model for {self.model_name}")

        self.tokenizer = model_tokenizer_dict[self.model_name].from_pretrained(
            self.model_weights, cache_dir=self.cache_dir
        )

    @staticmethod
    def required_packages() -> List[Text]:
        """Returns the extra python dependencies required."""
        return ["transformers"]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LanguageModelTokenizer":
        """Creates a new component (see parent class for full docstring)."""
        # Path to the dictionaries on the local filesystem.
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        words = self.tokenizer.tokenize(text)

        tokens = self._convert_words_to_tokens(words, text)

        return self._apply_token_pattern(tokens)
