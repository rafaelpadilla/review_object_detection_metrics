import input
import classes
import settings
import datasets
import confusion_matrix


def init(data, state):
    input.init(data, state)
    classes.init(data, state)
    datasets.init(data, state)
    settings.init(data, state)
    confusion_matrix.init(data, state)
    pass
