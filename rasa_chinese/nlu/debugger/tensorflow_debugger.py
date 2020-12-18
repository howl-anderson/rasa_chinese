import logging
import os
import typing
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn_crfsuite


class TensorflowDebugger(EntityExtractor):
    name = "addons_tf_debugger"

    provides = ["entities"]

    requires = ["addons_tf_input_fn", "addons_tf_input_meta"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None):

        super(TensorflowDebugger, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        return ["tensorflow", "seq2annotation"]

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        # read data according configure
        train_data_generator_func = kwargs.get('addons_tf_input_fn')
        corpus_meta_data = kwargs.get('addons_tf_input_meta')

        for data in train_data_generator_func():
            print(data)

        for data in train_data_generator_func():
            print(data)

        print(corpus_meta_data)

        # from seq2annotation.input import build_input_func
        # from seq2annotation.model import Model
        #
        # raw_config = {}
        # model = Model(raw_config)
        #
        # config = model.get_default_config()
        # config.update(raw_config)
        #
        # # task_status = TaskStatus(config)
        #
        # # read data according configure
        # train_data_generator_func = kwargs.get('addons_tf_input_fn')
        # corpus_meta_data = kwargs.get('addons_tf_input_meta')
        #
        # from seq2annotation.input import generate_tagset
        #
        # config['tags_data'] = generate_tagset(corpus_meta_data['tags'])
        #
        # # build model according configure
        #
        # # send START status to monitor system
        # # task_status.send_status(task_status.START)
        #
        # # train and evaluate model
        # train_input_func = build_input_func(train_data_generator_func, config)
        #
        # import tensorflow as tf
        #
        # data = train_input_func()
        #
        # with tf.Session() as sess:
        #     print(sess.run(data))

