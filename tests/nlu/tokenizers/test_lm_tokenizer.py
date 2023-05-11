import pytest
from typing import Dict, Optional

import rasa.shared.utils.io
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT, ACTION_TEXT, ACTION_NAME
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa_chinese.nlu.tokenizers.lm_tokenizer import LanguageModelTokenizer


def create_lm_tokenizer(config: Optional[Dict] = None) -> LanguageModelTokenizer:
    config = config if config else {}
    return LanguageModelTokenizer({**LanguageModelTokenizer.get_default_config(), **config})


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "我想去吃兰州拉面",  # easy/normal case
            ["我", "想", "去", "吃", "兰", "州", "拉", "面"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
        ),
        (
            "从东畈村走了。",  # OOV case: `畈` is a OOV word; TODO: not OOV anymore, fix it
            ["从", "东", "畈", "村", "走", "了", "。"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
        ),
        (
            "Micheal 你好吗？",  # Chinese mixed up with English
            ["Micheal", "你", "好", "吗", "？"],
            [(0, 7), (8, 9,), (9, 10), (10, 11), (11, 12)],
        ),
        (
            "我想买 iPhone 12 🤭",  # Chinese mixed up with English, numbers, and emoji
            ["我", "想", "买", "iPhone", "12", "🤭"],
            [(0, 1), (1, 2), (2, 3), (4, 10), (11, 13), (14, 15)],
        ),
    ],
)
def test_lm(text, expected_tokens, expected_indices):

    tk = create_lm_tokenizer()

    tokens = tk.tokenize(Message.build(text=text), attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]

