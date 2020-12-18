import functools
import logging
import typing
from typing import Any, Dict, List, Optional, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.utils.train_utils import override_defaults
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class TensorflowNLP(Component):
    name = 'addons_nlp_tf'

    provides = ['addons_tf_input_fn', 'addons_tf_input_meta']

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:

        self.input_func = None
        self.tag_list = None
        super(TensorflowNLP, self).__init__(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow", "tokenizer_tools"]

    @staticmethod
    def _turn_training_data_to_offset(training_data):
        from tokenizer_tools.tagset.offset.sequence import Sequence
        from tokenizer_tools.tagset.offset.span import Span
        from tokenizer_tools.tagset.offset.span_set import SpanSet

        for example in training_data.training_examples:
            span_set = SpanSet()

            text = [i for i in example.text]  # need to be str list (not str)
            intent = example.get("intent")

            for ent in example.get("entities", []):
                start, end, entity = ent["start"], ent["end"], ent["entity"]

                span_set.append(Span(start, end, entity))

            seq = Sequence(text, span_set, label=intent)

            yield seq

    @staticmethod
    def _collect_entity(training_data):
        entity_list = set()
        for example in training_data.training_examples:
            for ent in example.get("entities", []):
                entity_list.add(ent["entity"])

        return list(entity_list)

    @staticmethod
    def _collect_intent(training_data):
        intent_list = set()
        for example in training_data.training_examples:
            intent_list.add(example.data["intent"])

        return list(intent_list)

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        return {
            'addons_tf_input_fn': functools.partial(self._turn_training_data_to_offset, training_data),
            'addons_tf_input_meta': {
                'tags': self._collect_entity(training_data),
                'label': self._collect_intent(training_data)
            }
        }
