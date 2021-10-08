import supervisely_lib as sly
import globals as g
import ui_utils
import metrics
from widgets.confusion_matrix import ConfusionMatrix
from widgets.compare_gallery import CompareGallery
from widgets.sly_table import SlyTable

# from supervisely_lib.app.widgets.confusion_matrix import ConfusionMatrix

confusion_matrix = ConfusionMatrix(api=g.api, task_id=g.task_id, v_model='data.slyConfusionMatrix')
cm_image_table = SlyTable(g.api, g.task_id, 'data.CMTableImages', g.image_columns)
gallery_conf_matrix = CompareGallery(g.task_id, g.api, 'data.CMGallery', g.aggregated_meta)


def init(data, state):
    data['GalleryTable'] = {}
    data['CMNotification'] = {
        "options": {
            "name": "Select cell in confusion matrix to preview images with original annotations and detections.",
            "type": "info"
        }
    }
    state['activeFigure'] = None
    state['selection'] = {}
    state['selected'] = {'rowClass': None, 'colClass': None}
    state['CMActiveStep'] = None

    state["CMCollapsed1"] = True
    state["CMDisabled1"] = True
    state["CMShow1"] = False

    state["CMCollapsed2"] = True
    state["CMDisabled2"] = True
    state["CMShow2"] = False

    state["CMCollapsed3"] = True
    state["CMDisabled3"] = True
    state["CMShow3"] = False

    state['selectedRow'] = None

    data['CMImageTableTitle'] = "Statistic image table"

    conf_matrx_columns_v2 = []
    diagonal_max = 0
    max_value = 0
    cm_data = []

    slyConfusionMatrix = {
        "classes": conf_matrx_columns_v2,
        "diagonalMax": diagonal_max,
        "maxValue": max_value,
        "data": cm_data
    }
    data['slyConfusionMatrix'] = slyConfusionMatrix
    data['CMTableImages'] = {}
    data['CMGallery'] = {}
    data['CMImageTableDescription'] = "Please, select cell from confusion matrix."
    data['CMGalleryTitle'] = 'Please, select row from image statistic table.'


def reset_cm_state_to_default(api, task_id):
    fields = [
        {"field": "state.CMCollapsed1", "payload": True},
        {"field": "state.CMDisabled1", "payload": True},
        {"field": "state.CMShow1", "payload": False},

        {"field": "state.CMCollapsed2", "payload": True},
        {"field": "state.CMDisabled2", "payload": True},
        {"field": "state.CMShow2", "payload": False},

        {"field": "state.CMCollapsed3", "payload": True},
        {"field": "state.CMDisabled3", "payload": True},
        {"field": "state.CMShow3", "payload": False},

        {"field": "state.CMActiveStep", "payload": 0},

        # {"field": "data.CMImageTableTitle", "payload": 'Cell is not selected.'},
        {"field": "data.CMImageTableDescription1", "payload": 'Please, select cell from confusion matrix.'},
        {"field": "data.CMGalleryTitle", "payload": 'Please, select row from image statistic table.'},
    ]
    api.app.set_fields(task_id, fields)


# callback for CM image table
@g.my_app.callback("show_image_table")
@sly.timeit
def show_image_table(api: sly.Api, task_id, context, state, app_logger):
    v_model = "data.CMTableImages"

    fields = [
        # {"field": "state.CMActiveStep", "payload": 2},
        # {"field": "state.CMCollapsed2", "payload": False},
        # {"field": "state.CMDisabled2", "payload": False},
        {"field": "state.CMShow2", "payload": True},
        # {"field": "state.CMCollapsed3", "payload": True},
        # {"field": "state.CMDisabled3", "payload": True},
        {"field": "state.CMShow3", "payload": False},

        {"field": "data.CMImageTableDescription1", "payload": 'Please, select cell from confusion matrix.'},
        {"field": "data.CMGalleryTitle", "payload": 'Please, select row from image statistic table.'},
        {"field": "state.CMActiveNames", "payload": ['confusion_matrix', 'image_stat_table']},
    ]
    api.app.set_fields(task_id, fields)
    # print('selected row =', state['selected']['rowClass'])
    # print('selected col =', state['selected']['colClass'])
    ui_utils.show_image_table_body(api, task_id, state, v_model, cm_image_table)


@g.my_app.callback("show_images_gallery")
@sly.timeit
def show_images_gallery(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        # {"field": "state.CMActiveStep", "payload": 3},
        # {"field": "state.CMCollapsed3", "payload": False},
        # {"field": "state.CMDisabled3", "payload": False},
        {"field": "state.CMShow3", "payload": True},
        {"field": "state.CMActiveNames", "payload": ['confusion_matrix', 'image_stat_table', 'grid_gallery']},
    ]
    api.app.set_fields(task_id, fields)
    ui_utils.show_images_body(api, task_id, state, gallery_conf_matrix, "data.CMGalleryTitle")
