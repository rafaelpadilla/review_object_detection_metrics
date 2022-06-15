import supervisely as sly
import globals as g

import input
import datasets
import classes
import settings
import metrics


# input    1
# datasets 2
# classes  3
# settings 4
# metrics  5

def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None

    input.init(data, state)
    datasets.init(data, state)
    classes.init(data, state)
    settings.init(data, state)
    metrics.init(data, state, reconstruct=True)


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}
    if restart_from_step == 1:
        input.restart(data, state)

    if restart_from_step <= 2:
        if restart_from_step == 2:
            datasets.restart(data, state)
        else:
            datasets.init(data, state)

    if restart_from_step <= 3:
        if restart_from_step == 3:
            classes.restart(data, state)
        else:
            classes.init(data, state)

    if restart_from_step <= 4:
        if restart_from_step == 4:
            settings.restart(data, state)
        else:
            settings.init(data, state)

    if restart_from_step <= 5:
        metrics.init(data, state)

        # if restart_from_step == 5:
        #     metrics.restart(data, state)
        # else:
        #     metrics.init(data, state)

    fields = [
        {"field": "data", "payload": data, "append": True, "recursive": False},
        {"field": "state", "payload": state, "append": True, "recursive": False},
        {"field": "state.restartFrom", "payload": None},
        # {"field": f"state.collapsed{restart_from_step}", "payload": False},
        # {"field": f"state.disabled{restart_from_step}", "payload": False},
        {"field": "state.activeStep", "payload": restart_from_step},
    ]
    g.api.app.set_fields(g.task_id, fields)
    g.api.app.set_field(task_id, "data.scrollIntoView", f"step{restart_from_step}")


# @g.my_app.callback("restart")
# @sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
# def restart(api: sly.Api, task_id, context, state, app_logger):
#     restart_from_step = state["restartFrom"]
#     data = {}
#     state = {}
#
#     if restart_from_step == 1:
#         input.restart(data, state)
#     if restart_from_step <= 3:
#         classes.restart(data, state)
#     if restart_from_step <= 4:
#         settings.restart(data, state)
#
#     # if restart_from_step <= 3:
#     #     if restart_from_step == 3:
#     #         tags.restart(data, state)
#     #     else:
#     #         tags.init(data, state)
#     # if restart_from_step <= 4:
#     #     validate_training_data.init(data, state)
#     # if restart_from_step <= 5:
#     #     if restart_from_step == 5:
#     #         augs.restart(data, state)
#     #     else:
#     #         augs.init(data, state)
#     # if restart_from_step <= 6:
#     #     if restart_from_step == 6:
#     #         model_architectures.restart(data, state)
#     #     else:
#     #         model_architectures.init(data, state)
#     # if restart_from_step <= 7:
#     #     if restart_from_step == 7:
#     #         hyperparameters.restart(data, state)
#     #     else:
#     #         hyperparameters.init(data, state)
#     # if restart_from_step <= 8:
#     #     if restart_from_step == 8:
#     #         hyperparameters_python.restart(data, state)
#     #     else:
#     #         hyperparameters_python.init(data, state)
#     #
#     fields = [
#         {"field": "data", "payload": data, "append": True, "recursive": False},
#         {"field": "state", "payload": state, "append": True, "recursive": False},
#         {"field": "state.restartFrom", "payload": None},
#         {"field": f"state.collapsed{restart_from_step}", "payload": False},
#         {"field": f"state.disabled{restart_from_step}", "payload": False},
#         {"field": "state.activeStep", "payload": restart_from_step},
#     ]
#     g.api.app.set_fields(g.task_id, fields)
#     g.api.app.set_field(task_id, "data.scrollIntoView", f"step{restart_from_step}")




