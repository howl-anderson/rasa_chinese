import logging
import os
import shutil
import tempfile
import typing
from typing import Any, Dict, Optional, Text

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__name__)


class BilstmCrfPaddleEntityExtractor(EntityExtractor):
    name = "addons_ner_bilstm_crf_paddle"

    provides = ["entities"]

    requires = ["addons_paddle_input_fn", "addons_paddle_input_meta"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 model_dir=None) -> None:

        self.result_dir = None if 'result_dir' not in component_config else \
        component_config['result_dir']

        self.predict_fn = None
        self.model_dir = model_dir

        super(BilstmCrfPaddleEntityExtractor, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        return ["paddle", "seq2annotation"]

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        from seq2annotation.trainer.paddle_train import Train

        raw_config = self.component_config

        print(raw_config)

        if 'result_dir' not in raw_config:
            raw_config['result_dir'] = tempfile.mkdtemp()

        # read data according configure
        raw_config['data_source_scheme'] = 'raw'
        raw_config['corpus_train_input_func'] = kwargs.get(
            'addons_paddle_input_fn')
        raw_config['corpus_eval_input_func'] = None
        raw_config['corpus_meta_info'] = kwargs.get('addons_paddle_input_meta')

        train = Train()
        final_saved_model = train.train(addition_config=raw_config,
                                        return_empty=True)

        self.result_dir = final_saved_model

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any
    ) -> "Component":
        if cached_component:
            return cached_component
        else:
            return cls(meta, model_dir)

    def process(self, message: Message, **kwargs: Any) -> None:
        from seq2annotation.server.paddle_inference import Inference

        real_result_dir = os.path.join(self.model_dir, self.result_dir)
        print(real_result_dir)

        # for cache
        if not self.predict_fn:
            self.predict_fn = Inference(real_result_dir)

        input_text = message.text

        seq = self.predict_fn.infer(input_text)

        seq.span_set.fill_text(input_text)
        entity_set = []
        for span in seq.span_set:
            ent = {
                "entity": span.entity,
                "value": span.value,
                "start": span.start,
                "confidence": None,
                "end": span.end
            }
            entity_set.append(ent)

        extracted = self.add_extractor_name(entity_set)

        message.set("entities",
                    message.get("entities", []) + extracted,
                    add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        print(model_dir)
        saved_model_dir = os.path.join(model_dir, self.name)

        print(saved_model_dir)
        print(self.result_dir)

        shutil.copytree(self.result_dir, saved_model_dir)

        return {'result_dir': self.name}
