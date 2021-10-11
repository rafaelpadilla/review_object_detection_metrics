import supervisely_lib as sly
import datasets as ds
import globals as g
import ui
import metrics
import download_data as dd
from sly_progress import init_progress


def init(data, state):

    total_img_num = 0
    for k, v in ds.image_dict['gt_images'].items():
        total_img_num += len(v)

    data['totalImagesCount'] = total_img_num
    data['doneSettings'] = False

    state['GlobalSettingsCollapsed'] = True
    state['GlobalSettingsDisabled'] = True
    state['GlobalShowSettings'] = False
    state['GlobalSettingsLoaded'] = False

    state["samplePercent"] = 5
    state['IoUThreshold'] = 45
    state['ScoreThreshold'] = 25

    state['settingsLoading'] = False

    state['DownLoadAnnotations'] = False

    '''     {{state.progressDownLoadAnnotations}}: {{state.progressCurrentDownLoadAnnotations}} / {{state.progressTotalDownLoadAnnotations}}
        </div>
        <el-progress :percentage="data.progressPercentUploadDir"></el-progress>
    </div>'''


def restart(data, state):
    state['GlobalActiveStep'] = 4

    state['GlobalSettingsCollapsed'] = False
    state['GlobalSettingsDisabled'] = False
    state['GlobalMetricsCollapsed'] = True
    state['GlobalMetricsDisabled'] = True
    data['doneSettings'] = False
    metrics.init(data, state)


@g.my_app.callback("evaluate_button_click")
@sly.timeit
def evaluate_button_click(api: sly.Api, task_id, context, state, app_logger):
    global cm, object_mapper, filtered_confidences, gts, pred  #  , dataset_names, previous_percentage
    selected_classes = state['selectedClasses']
    percentage = state['samplePercent']
    iou_threshold = state['IoUThreshold'] / 100
    score_threshold = state['ScoreThreshold'] / 100

    metrics.confusion_matrix.reset_cm_state_to_default(api, task_id)
    fields = [
        {"field": "state.loading", "payload": True},

        {"field": "state.GlobalActiveStep", "payload": 5},
        {"field": "state.GlobalClassesCollapsed", "payload": True},
        {"field": "state.GlobalMetricsCollapsed", "payload": True},
        {"field": "state.GlobalMetricsDisabled", "payload": True},

        {"field": "state.CMShow1", "payload": False},
        {"field": "state.CMShow2", "payload": False},
        {"field": "state.CMShow3", "payload": False},
        {"field": "state.CMActiveNames", "payload": []},
        {"field": "data.CMImageTableDescription1", "payload": 'Please, select cell from Confusion Matrix.'},
        {"field": "data.CMImageTableDescription2", "payload": None},
        {"field": "data.CMImageTableDescription3", "payload": None},

        {"field": "state.perClassShow1", "payload": False},
        {"field": "state.perClassShow2", "payload": False},
        {"field": "state.perClassShow3", "payload": False},
        {"field": "state.perClassActiveNames", "payload": []},

    ]
    api.app.set_fields(task_id, fields)

    if selected_classes:
        filtered_confidences = dd.download_and_prepare_data(selected_classes,
                                                            percentage=percentage,
                                                            confidence_threshold=score_threshold)

        if filtered_confidences['gt_images'] and filtered_confidences['pred_images']:
            metrics.confusion_matrix.confusion_matrix.set_data(gt=filtered_confidences['gt_images'],
                                                               det=filtered_confidences['pred_images'])
            metrics.confusion_matrix.confusion_matrix.reset_thresholds(iou_threshold=iou_threshold,
                                                                       score_threshold=score_threshold)
            metrics.confusion_matrix.confusion_matrix.update()
            object_mapper = metrics.confusion_matrix.confusion_matrix.object_maps
            cm = metrics.confusion_matrix.confusion_matrix.cm_dict

            gts, pred, dataset_names = dd.get_prepared_data(api=g.api,
                                                            src_list=filtered_confidences['gt_images'],
                                                            dst_list=filtered_confidences['pred_images'],
                                                            encoder=dd.RepoBoundingBox)
            method = metrics.MethodAveragePrecision.EVERY_POINT_INTERPOLATION

            metrics.overall_metrics.calculate_overall_metrics(api, task_id, gts, pred, g.pred_project_info.name, method,
                                                              iou_threshold, score_threshold)
            metrics.per_image_metrics.calculate_per_image_metrics(api, task_id, gts, pred, method,
                                                                  iou_threshold, score_threshold)
            metrics.per_class_metrics.calculate_per_classes_metrics(api, task_id, gts, pred, g.pred_project_info.name,
                                                                    method,
                                                                    iou_threshold, score_threshold)
    fields = [
        {"field": "data.doneSettings", "payload": True},

        {"field": "state.loading", "payload": False},

        {"field": "state.GlobalActiveStep", "payload": 5},
        {"field": "state.GlobalClassesCollapsed", "payload": True},
        {"field": "state.GlobalSettingsCollapsed", "payload": True},
        {"field": "state.GlobalMetricsCollapsed", "payload": False},
        {"field": "state.GlobalMetricsDisabled", "payload": False},

        {"field": "state.CMShow1", "payload": True},
        {"field": "state.CMActiveNames", "payload": ['confusion_matrix']},

        {"field": "state.perClassShow1", "payload": True},
        {"field": "state.perClassActiveNames", "payload": ['per_class_table']},
    ]
    api.app.set_fields(task_id, fields)


cm = {}
object_mapper = {}
gts = {}
pred = {}
filtered_confidences = {}
