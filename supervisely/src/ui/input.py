import supervisely_lib as sly
import globals as g
import datasets
import classes
import settings
import metrics


def init(data, state):
    state['GlobalInputCollapsed'] = False
    state['GlobalInputDisabled'] = False
    state['doneInput'] = False
    state['InputInProgress'] = True

    # data["gtProjectId"] = g.gt_project_info.id
    # data["gtProjectName"] = g.gt_project_info.name
    # data["gtProjectPreviewUrl"] = g.api.image.preview_url(g.gt_project_info.reference_image_url, 100, 100)
    #
    # data["predProjectId"] = g.pred_project_info.id
    # data["predProjectName"] = g.pred_project_info.name
    # data["predProjectPreviewUrl"] = g.api.image.preview_url(g.pred_project_info.reference_image_url, 100, 100)


def restart(data, state):
    state['GlobalInputCollapsed'] = False
    state['GlobalInputDisabled'] = False
    state['doneInput'] = False
    state['InputInProgress'] = True

    datasets.init(data, state)
    classes.init(data, state)
    settings.init(data, state)
    metrics.init(data, state, reconstruct=False)
    state['GlobalActiveStep'] = 1


@g.my_app.callback("set_projects")
@sly.timeit
def set_projects(api: sly.Api, task_id, context, state, app_logger):

    gt_project_info = api.project.get_info_by_id(state['gtProjectId'], raise_error=True)
    pr_project_info = api.project.get_info_by_id(state['predProjectId'], raise_error=True)

    fields = [
        {"field": "state.GlobalInputCollapsed", "payload": True},
        {"field": "state.GlobalInputDisabled", "payload": False},
        {"field": "state.doneInput", "payload": True},
        {"field": "state.InputInProgress", "payload": False},

        {"field": "state.GlobalDatasetsCollapsed", "payload": False},
        {"field": "state.GlobalDatasetsDisabled", "payload": False},
        {"field": "state.doneDatasets", "payload": False},
        {"field": "state.DatasetsInProgress", "payload": True},

        {"field": "state.GlobalActiveStep", "payload": 2},
        {"field": "data.gtProjectId", "payload": gt_project_info.id},
        {"field": "data.gtProjectName", "payload": gt_project_info.name},
        {"field": "data.gtProjectPreviewUrl",
         "payload": g.api.image.preview_url(gt_project_info.reference_image_url, 100, 100)},

        {"field": "data.predProjectId", "payload": pr_project_info.id},
        {"field": "data.predProjectName", "payload": pr_project_info.name},
        {"field": "data.predProjectPreviewUrl",
         "payload": g.api.image.preview_url(pr_project_info.reference_image_url, 100, 100)},

        {"field": "state.doneInput", "payload": True},

    ]
    api.app.set_fields(task_id, fields)
