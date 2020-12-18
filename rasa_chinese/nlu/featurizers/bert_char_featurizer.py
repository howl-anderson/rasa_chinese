from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from rasa_chinese.nlu.featurizers.bert_base import BertBase

logger = logging.getLogger(__name__)


class BertCharFeaturizer(BertBase):
    provides = ["char_features"]
    name = "bert_char_featurizer"

    @staticmethod
    def _combine_with_existing_text_features(message, additional_features):
        if message.get("char_features") is not None:
            return np.hstack((message.get("char_features"), additional_features))
        else:
            return additional_features

    def _set_feature(self, example, feature):
        text_length = len(example.text)

        # [CLS] X1 X2 X3 [SEP] ...
        feature = feature[1: text_length + 1]

        new_feature = self._combine_with_existing_text_features(example, feature)

        example.set(
            "char_features",
            new_feature
        )
