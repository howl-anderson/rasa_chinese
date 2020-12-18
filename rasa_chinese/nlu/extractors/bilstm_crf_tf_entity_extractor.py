import logging
import os
import shutil
import tempfile
import typing
from typing import Any, Dict, List, Optional, Text, Tuple, Callable

from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


if typing.TYPE_CHECKING:
    from tokenizer_tools.tagset.offset.sequence import Sequence
    import sklearn_crfsuite

logger = logging.getLogger(__name__)


class BilstmCrfTensorFlowEntityExtractor(EntityExtractor):
    name = "addons_ner_bilstm_crf_tf"

    provides = ["entities"]

    requires = ["addons_tf_input_fn", "addons_tf_input_meta"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 model_dir=None,
                 predict_fn: Optional[Callable] = None) -> None:

        self.result_dir = None if 'result_dir' not in component_config else component_config['result_dir']

        self.predict_fn = predict_fn
        self.model_dir = model_dir

        super(BilstmCrfTensorFlowEntityExtractor, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        return ["tensorflow", "seq2annotation"]

    def _keras_data_preprocss(self, data: 'List[Sequence]', tag_lookuper, maxlen=None):
        import tensorflow as tf
        from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo

        raw_x = []
        raw_y = []

        for intent_data in data:
            offset_data = intent_data
            tags = offset_to_biluo(offset_data)
            words = offset_data.text

            tag_ids = [tag_lookuper.lookup(i) for i in tags]
            word_ids = ''.join(words)

            raw_x.append(word_ids)
            raw_y.append(tag_ids)

        if maxlen is None:
            maxlen = max(len(s) for s in raw_x)

        x = get_np_feature(raw_x, maxlen)

        y = tf.keras.preprocessing.sequence.pad_sequences(raw_y, maxlen, value=0, padding='post')

        return x, y

    def _keras_train(
        self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        from tensorflow.python.keras.layers import Input, Masking
        from tensorflow.python.keras.models import Sequential
        from tf_crf_layer.layer import CRF
        from tf_crf_layer.loss import crf_loss
        from tf_crf_layer.metrics import crf_accuracy
        from seq2annotation.input import generate_tagset
        from seq2annotation.input import build_input_func
        from seq2annotation.input import Lookuper

        config = self.component_config

        if 'result_dir' not in config:
            config['result_dir'] = tempfile.mkdtemp()

        # read data according configure
        train_data_generator_func = kwargs.get('addons_tf_input_fn')
        corpus_meta_data = kwargs.get('addons_tf_input_meta')

        config['tags_data'] = generate_tagset(corpus_meta_data['tags'])

        # train and evaluate model
        train_input_func = build_input_func(train_data_generator_func, config)

        tag_lookuper = Lookuper({v: i for i, v in enumerate(config['tags_data'])})

        maxlen = 25

        offset_data = train_input_func()
        train_x, train_y = self._keras_data_preprocss(offset_data, tag_lookuper, maxlen)

        EPOCHS = 1

        tag_size = tag_lookuper.size()

        model = Sequential()
        model.add(Input(shape=(25, 768)))
        model.add(Masking())
        model.add(CRF(tag_size))
        model.compile('adam', loss=crf_loss)
        model.summary()

        model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
        model.fit(train_x, train_y, epochs=EPOCHS)

    def train(
        self, training_data: TrainingData, cfg: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        from seq2annotation.input import generate_tagset
        from seq2annotation.input import build_input_func
        from seq2annotation.model import Model

        raw_config = self.component_config

        print(raw_config)

        if 'result_dir' not in raw_config:
            raw_config['result_dir'] = tempfile.mkdtemp()

        model = Model(raw_config)

        config = model.get_default_config()
        config.update(raw_config)

        # task_status = TaskStatus(config)

        # read data according configure
        train_data_generator_func = kwargs.get('addons_tf_input_fn')
        corpus_meta_data = kwargs.get('addons_tf_input_meta')

        config['tags_data'] = generate_tagset(corpus_meta_data['tags'])

        # build model according configure

        # send START status to monitor system
        # task_status.send_status(task_status.START)

        # train and evaluate model
        train_input_func = build_input_func(train_data_generator_func, config)

        evaluate_result, export_results, final_saved_model = model.train_and_eval_then_save(
            train_input_func,
            None,
            config
        )

        # task_status.send_status(task_status.DONE)

        self.result_dir = final_saved_model

    @classmethod
    def _keras_load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any
    ) -> "Component":
        pass

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
            from tensorflow.contrib import predictor

            real_result_dir = os.path.join(model_dir, meta['result_dir'])

            predict_fn = predictor.from_saved_model(real_result_dir)
            return cls(meta, model_dir, predict_fn)

    def process(self, message: Message, **kwargs: Any) -> None:
        from tokenizer_tools.tagset.NER.BILUO import BILUOSequenceEncoderDecoder
        from tokenizer_tools.tagset.offset.sequence import Sequence

        decoder = BILUOSequenceEncoderDecoder()

        real_result_dir = os.path.join(self.model_dir, self.result_dir)
        print(real_result_dir)

        input_text = message.text

        input_feature = {
            'words': [[i for i in input_text]],
            'words_len': [len(input_text)],
        }

        print(input_feature)

        predictions = self.predict_fn(input_feature)
        tags = predictions['tags'][0]
        # print(predictions['tags'])

        # decode Unicode
        tags_seq = [i.decode() for i in tags]

        print(tags_seq)

        # BILUO to offset
        failed = False
        try:
            seq = decoder.to_offset(tags_seq, input_text)
        except Exception as e:
            # invalid tag sequence will raise exception
            # so return a empty result
            logger.error("Decode error: {}".format(e))
            seq = Sequence(input_text)
            failed = True
        # print(seq)

        print(seq, tags_seq, failed)

        entity_set = []

        seq.span_set.fill_text(input_text)

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
