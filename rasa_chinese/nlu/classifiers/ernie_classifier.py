from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.extractors.extractor import EntityExtractor


class ERNIEClassifier(IntentClassifier, EntityExtractor):
    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    supported_language_list = None

    # Defines what language(s) this component can NOT handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    not_supported_language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:

        if not component_config:
            component_config = {}

        # makes sure the name of the configuration is part of the config
        # this is important for e.g. persistence
        component_config["name"] = self.name

        self.component_config = override_defaults(self.defaults, component_config)

        self.partial_processing_pipeline = None
        self.partial_processing_context = None

    @classmethod
    def required_packages(cls) -> List[Text]:
        """Specify which python packages need to be installed.
        E.g. ``["spacy"]``. More specifically, these should be
        importable python package names e.g. `sklearn` and not package
        names in the dependencies sense e.g. `scikit-learn`
        This list of requirements allows us to fail early during training
        if a required package is not installed.
        Returns:
            The list of required package names.
        """

        return ["paddle-ernie", "paddlepaddle"]

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file.
        After a component has been trained, it will be persisted by
        calling `persist`. When the pipeline gets loaded again,
        this component needs to be able to restore itself.
        Components can rely on any context attributes that are
        created by :meth:`components.Component.create`
        calls to components previous to this one.
        Args:
            meta: Any configuration parameter related to the model.
            model_dir: The directory to load the component from.
            model_metadata: The model's :class:`rasa.nlu.model.Metadata`.
            cached_component: The cached component.
        Returns:
            the loaded component
        """

        if cached_component:
            return cached_component

        return cls(meta)

    @classmethod
    def create(
        cls, component_config: Dict[Text, Any], config: RasaNLUModelConfig
    ) -> "Component":
        """Creates this component (e.g. before a training is started).
        Method can access all configuration parameters.
        Args:
            component_config: The components configuration parameters.
            config: The model configuration parameters.
        Returns:
            The created component.
        """

        # Check language supporting
        language = config.language
        if not cls.can_handle_language(language):
            # check failed
            raise UnsupportedLanguageError(cls.name, language)

        return cls(component_config)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component.
        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.train`
        of components previous to this one.
        Args:
            training_data:
                The :class:`rasa.nlu.training_data.training_data.TrainingData`.
            config: The model configuration parameters.
        """

        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.
        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.
        Args:
            message: The :class:`rasa.nlu.training_data.message.Message` to process.
        """

        pass

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading.
        Args:
            file_name: The file name of the model.
            model_dir: The directory to store the model to.
        Returns:
            An optional dictionary with any information about the stored model.
        """

        pass

    @classmethod
    def cache_key(
        cls, component_meta: Dict[Text, Any], model_metadata: "Metadata"
    ) -> Optional[Text]:
        """This key is used to cache components.
        If a component is unique to a model it should return None.
        Otherwise, an instantiation of the
        component will be reused for all models where the
        metadata creates the same key.
        Args:
            component_meta: The component configuration.
            model_metadata: The component's :class:`rasa.nlu.model.Metadata`.
        Returns:
            A unique caching key.
        """

        return None

    def __getstate__(self) -> Any:
        d = self.__dict__.copy()
        # these properties should not be pickled
        if "partial_processing_context" in d:
            del d["partial_processing_context"]
        if "partial_processing_pipeline" in d:
            del d["partial_processing_pipeline"]
        return d

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__

    def prepare_partial_processing(
        self, pipeline: List["Component"], context: Dict[Text, Any]
    ) -> None:
        """Sets the pipeline and context used for partial processing.
        The pipeline should be a list of components that are
        previous to this one in the pipeline and
        have already finished their training (and can therefore
        be safely used to process messages).
        Args:
            pipeline: The list of components.
            context: The context of processing.
        """

        self.partial_processing_pipeline = pipeline
        self.partial_processing_context = context

    def partially_process(self, message: Message) -> Message:
        """Allows the component to process messages during
        training (e.g. external training data).
        The passed message will be processed by all components
        previous to this one in the pipeline.
        Args:
            message: The :class:`rasa.nlu.training_data.message.Message` to process.
        Returns:
            The processed :class:`rasa.nlu.training_data.message.Message`.
        """

        if self.partial_processing_context is not None:
            for component in self.partial_processing_pipeline:
                component.process(message, **self.partial_processing_context)
        else:
            logger.info("Failed to run partial processing due to missing pipeline.")
        return message
