import supervisely_lib as sly
import globals as g


def init(data, state):
    data["gtProjectId"] = g.gt_project_info.id
    data["gtProjectName"] = g.gt_project_info.name
    data["gtProjectPreviewUrl"] = g.api.image.preview_url(g.gt_project_info.reference_image_url, 100, 100)

    data["predProjectId"] = g.pred_project_info.id
    data["predProjectName"] = g.pred_project_info.name
    data["predProjectPreviewUrl"] = g.api.image.preview_url(g.pred_project_info.reference_image_url, 100, 100)