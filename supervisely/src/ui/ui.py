
import globals as g

import input
import datasets
import classes
import settings
import metrics


def init(data, state):
    input.init(data, state)
    classes.init(data, state)
    datasets.init(data, state)
    settings.init(data, state)
    metrics.init(data, state)

    # for collapses
    state['activeName'] = "Classes"
    # for stepper
    state['GlobalActiveStep'] = 1
    state['GlobalClassesCollapsed'] = False
    state['GlobalClassesDisabled'] = False
    state['GlobalSettingsCollapsed'] = True
    state['GlobalSettingsDisabled'] = True
    state['GlobalMetricsCollapsed'] = True
    state['GlobalMetricsDisabled'] = True






