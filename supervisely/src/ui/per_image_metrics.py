# from supervisely_lib.app.widgets.sly_table import SlyTable
from widgets.sly_table import SlyTable
import globals as g
import metrics

image_sly_table = SlyTable(g.api, g.task_id, "data.perImageTable", metrics.image_columns)


def init(data, state):
    data['perImageTable'] = {}
    data['perImage'] = {}
    data['perImageGalleryTitle'] = 'Please, select row from ImageTable.'


def calculate_per_image_metrics(api, task_id, gts, pred, method, iou_threshold, score_threshold):
    images_pd_data = metrics.calculate_image_mAP(gts, pred, method, target_class=None, iou=iou_threshold,
                                                 score=score_threshold, need_rez=False)
    image_sly_table.set_data(images_pd_data)
    image_sly_table.update()

