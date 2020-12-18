try:
    # form rasa 1.6.x
    from rasa.nlu.featurizers.featurizer import Featurizer
except ImportError:
    # for rasa 1.5.x
    from rasa.nlu.featurizers import Featurizer


class ContribFeaturizer(Featurizer):
    pass
