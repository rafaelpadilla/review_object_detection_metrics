import metrics


def init(data, state):
    data['perImageTableImages'] = {}
    data['perImagesPreviewContent'] = {}
    data['perImagesPreviewOptions'] = {}


def calculate_per_image_metrics(api, task_id, gts, pred, method, iou_threshold, score_threshold):
    images_pd_data = metrics.calculate_image_mAP(gts, pred, method, target_class=None, iou=iou_threshold,
                                                 score=score_threshold, need_rez=False)
    fields = [
        # {"field": "data.loading", "payload": False},
        {"field": "data.perImageTableImages", "payload": {"columns": metrics.image_columns, "data": images_pd_data}},
    ]
    api.app.set_fields(task_id, fields)
