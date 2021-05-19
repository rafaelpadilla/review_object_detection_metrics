import os
from pathlib import Path
import sys
import supervisely_lib as sly


my_app = sly.AppService()
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])

api: sly.Api = my_app.public_api
task_id = my_app.task_id

gt_project_id = int(os.environ['modal.state.gtProjectId'])
gt_project_info = api.project.get_info_by_id(gt_project_id, raise_error=True)
gt_meta = sly.ProjectMeta.from_json(api.project.get_meta(gt_project_id))

pred_project_id = int(os.environ['modal.state.predProjectId'])
pred_project_info = api.project.get_info_by_id(pred_project_id, raise_error=True)
pred_meta = sly.ProjectMeta.from_json(api.project.get_meta(pred_project_id))

# api.project.get_meta(pred_project_id)
# meta = app.public_api.project.get_meta(dst_project_id)

root_source_dir = str(Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"Source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")


