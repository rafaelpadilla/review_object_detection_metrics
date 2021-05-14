import supervisely_lib as sly


def init(data, state):
    state["samplePercent"] = 5
    state['IoUThreshold'] = 5
    state['ScoreThreshold'] = 45
