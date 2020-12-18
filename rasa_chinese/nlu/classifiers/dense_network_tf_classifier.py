import logging
import os
import shutil
import json
import tempfile
import typing
from typing import Any, Dict, Optional, Text

from rasa.nlu.components import Component
from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message


class DenseNetworkTensorFlowClassifier(Component):
    name = "addons_intent_classifier_textcnn_tf"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 model_dir=None) -> None:

        self.result_dir = None if 'result_dir' not in component_config else component_config['result_dir']
        self.lookup_table_file = None if 'lookup_table_file' not in component_config else component_config['lookup_table_file']
        self.epoch = component_config.get("epoch", 10)
        self.batch_size = component_config.get("batch_size", 32)

        self.predict_fn = None
        self.model_dir = model_dir
        self.lookup_table = None

        super(DenseNetworkTensorFlowClassifier, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        return ["tensorflow"]

    @staticmethod
    def build_model(feature_length, intent_number):
        import tensorflow as tf
        from tensorflow.keras import layers

        model = tf.keras.Sequential([
            # Adds a densely-connected layer with 64 units to the model:
            layers.Dense(64, activation='relu', input_shape=(feature_length,)),
            # Add another:
            # layers.Dense(64, activation='relu'),
            # Add a softmax layer with 10 output units:
            layers.Dense(intent_number, activation='softmax')])

        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def get_input_data(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig):
        import numpy as np

        whole_intent_text_set = set()

        intent_text_list = []
        text_feature_list = []
        for example in training_data.training_examples:
            text_feature = example.get('text_features')
            text_feature_list.append(text_feature)

            intent_text = example.get('intent')
            intent_text_list.append(intent_text)
            whole_intent_text_set.add(intent_text)

        intent_lookup_table = {value: index for index, value in enumerate(whole_intent_text_set)}
        intent_int_list = [intent_lookup_table[i] for i in intent_text_list]

        intent_np_array = np.array(intent_int_list)
        text_feature_np_array = np.array(text_feature_list)

        intent_number = len(whole_intent_text_set)
        feature_length = text_feature_np_array.shape[-1]

        return text_feature_np_array, intent_np_array, feature_length, intent_number, intent_lookup_table

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        import tensorflow as tf

        data, labels, feature_length, intent_number, intent_lookup_table = self.get_input_data(training_data, config)

        model = self.build_model(feature_length, intent_number)

        model.fit(data, labels, epochs=10, batch_size=32)

        final_saved_model = tempfile.TemporaryDirectory().name
        logger.debug(f"Temperary, SavedModel are store in {final_saved_model}")

        tf.keras.experimental.export_saved_model(model, final_saved_model)

        self.result_dir = final_saved_model
        self.lookup_table = intent_lookup_table

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
        import tensorflow as tf
        import numpy as np

        real_result_dir = os.path.join(self.model_dir, self.result_dir)
        # print(real_result_dir)

        if self.predict_fn is None:
            self.predict_fn = tf.keras.experimental.load_from_saved_model(real_result_dir)

        real_lookup_table_file = os.path.join(real_result_dir, self.lookup_table_file)
        # print(real_lookup_table_file)

        if self.lookup_table is None:
            with open(real_lookup_table_file, 'rt') as fd:
                self.lookup_table = json.load(fd)

        text_feature = message.get("text_features")
        np_feature = np.array([text_feature])

        predict_np_int = self.predict_fn.predict(np_feature)

        intent_score = []
        for intent_id, score in enumerate(predict_np_int[0]):
            # convert np.float32 to vanilla float,
            # if not it will cause json_dumps of ujson raise exception OverflowError: Maximum recursion level reached
            # see https://github.com/esnme/ultrajson/issues/221
            float_score = float(score)
            intent_score.append((float_score, intent_id))

        reversed_lookup_table = {index: value for value, index in self.lookup_table.items()}
        intent_str_score = [(k, reversed_lookup_table[v]) for k, v in intent_score]

        sorted_intent_str_score = sorted(intent_str_score, key=lambda x: x[0], reverse=True)

        # print(sorted_intent_str_score)

        self._set_intent_output(message, sorted_intent_str_score)

    def _set_intent_output(self, message, intent_score):
        first_candidate = intent_score[0]
        intent = {"name": first_candidate[1], "confidence": first_candidate[0]}

        intent_ranking = [{"name": name,
                           "confidence": score}
                          for score, name in intent_score]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        print(model_dir)
        saved_model_dir = os.path.join(model_dir, self.name)

        print(saved_model_dir)
        print(self.result_dir)

        shutil.copytree(self.result_dir, saved_model_dir)

        # serialize lookup table for intent string <-> intent id
        serialized_lookup_table_file = os.path.join(saved_model_dir, 'lookup_table.json')
        with open(serialized_lookup_table_file, 'wt') as fd:
            json.dump(self.lookup_table, fd)

        return {'result_dir': self.name, 'lookup_table_file': 'lookup_table.json', "epoch": self.epoch, "batch_size": self.batch_size}
