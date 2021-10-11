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

    state["GTteamId"] = g.team_id
    state["GTworkspaceId"] = g.workspace_id
    state["PRteamId"] = g.team_id
    state["PRworkspaceId"] = g.workspace_id
    state["gtProjectId"] = None
    state["predProjectId"] = None


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

    g.gt_project_info = api.project.get_info_by_id(state['gtProjectId'], raise_error=True)
    g.pr_project_info = api.project.get_info_by_id(state['predProjectId'], raise_error=True)

    fields = [
        {"field": "state.GlobalInputCollapsed", "payload": True},
        {"field": "state.GlobalInputDisabled", "payload": False},
        {"field": "state.doneInput", "payload": True},
        {"field": "state.InputInProgress", "payload": False},

        {"field": "state.GlobalDatasetsCollapsed", "payload": False},
        {"field": "state.GlobalDatasetsDisabled", "payload": False},
        {"field": "state.doneDatasets", "payload": False},

        {"field": "state.GlobalClassesCollapsed", "payload": True},
        {"field": "state.GlobalClassesDisabled", "payload": True},
        {"field": "state.doneClasses", "payload": False},

        {"field": "state.GlobalSettingsCollapsed", "payload": True},
        {"field": "state.GlobalSettingsDisabled", "payload": True},
        {"field": "state.doneDatasets", "payload": False},

        {"field": "state.GlobalActiveStep", "payload": 2},

        {"field": "state.gtProjectId", "payload": g.gt_project_info.id},
        {"field": "state.gtProjectName", "payload": g.gt_project_info.name},
        {"field": "state.gtProjectPreviewUrl",
         "payload": g.api.image.preview_url(g.gt_project_info.reference_image_url, 100, 100)},

        {"field": "state.predProjectId", "payload": g.pr_project_info.id},
        {"field": "state.predProjectName", "payload": g.pr_project_info.name},
        {"field": "state.predProjectPreviewUrl",
         "payload": g.api.image.preview_url(g.pr_project_info.reference_image_url, 100, 100)},

        {"field": "state.doneInput", "payload": True},

    ]
    api.app.set_fields(task_id, fields)
