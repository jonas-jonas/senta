from .senta import Senta # noqa


def load(nlp=None):
    return Senta(nlp)
