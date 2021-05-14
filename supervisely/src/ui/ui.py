import input
import classes
import settings
import datasets


def init(data, state):
    input.init(data, state)
    classes.init(data, state)
    datasets.init(data, state)
    settings.init(data, state)

    pass
