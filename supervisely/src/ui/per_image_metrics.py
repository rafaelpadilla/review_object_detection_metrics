import supervisely_lib as sly
import globals as g
from widgets.sly_table import SlyTable
from widgets.compare_gallery import CompareGallery

import metrics
import ui_utils
# from supervisely_lib.app.widgets.sly_table import SlyTable

image_sly_table = SlyTable(g.api, g.task_id, "data.perImageTable", g.image_columns)
gallery_per_image = CompareGallery(g.task_id, g.api, 'data.perImage', g.aggregated_meta)


def init(data, state):
    data['perImageTable'] = {}
    data['perImage'] = {}
    data['perImageGalleryTitle'] = 'Please, select row from ImageTable.'


def calculate_per_image_metrics(api, task_id, gts, pred, method, iou_threshold, score_threshold):
    images_pd_data = metrics.calculate_image_mAP(gts, pred, method, target_class=None, iou=iou_threshold,
                                                 score=score_threshold, need_rez=False)
    image_sly_table.set_data(images_pd_data)
    image_sly_table.update()


@g.my_app.callback("show_images_per_image")
@sly.timeit
def show_images_per_image(api: sly.Api, task_id, context, state, app_logger):
    ui_utils.show_images_body(api, task_id, state, gallery_per_image, "data.perImageGalleryTitle")
