import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import supervisely_lib as sly
from collections import namedtuple

# only for convenient debug
debug_env_path = os.path.join(root_source_dir, "supervisely/train", "debug.env")
secret_debug_env_path = os.path.join(root_source_dir, "supervisely/train", "secret_debug.env")
load_dotenv(debug_env_path)
load_dotenv(secret_debug_env_path, override=True)

my_app = sly.AppService()
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])

api: sly.Api = my_app.public_api
task_id = my_app.task_id

gt_project_id = int(os.environ['modal.state.gtProjectId'])
gt_project_info = api.project.get_info_by_id(gt_project_id, raise_error=True)
_gt_meta_ = api.project.get_meta(gt_project_id)
gt_meta = sly.ProjectMeta.from_json(_gt_meta_)

pred_project_id = int(os.environ['modal.state.predProjectId'])
pred_project_info = api.project.get_info_by_id(pred_project_id, raise_error=True)
_pred_meta_ = api.project.get_meta(pred_project_id)
pred_meta = sly.ProjectMeta.from_json(_pred_meta_)

# @TODO: get supervisely format meta to
aggregated_meta = {'classes': [], 'tags': _pred_meta_['tags'], 'projectType': 'images'}
for i in _gt_meta_['classes']:
    for j in _pred_meta_['classes']:
        if i['title'] == j['title'] and i['shape'] == j['shape']:
            aggregated_meta['classes'].append(i)

aggregated_meta = sly.ProjectMeta.from_json(aggregated_meta)

root_source_dir = str(Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"Source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")


result = namedtuple('Result', ['TP', 'FP', 'NPOS', 'Precision', 'Recall', 'AP'])
table_classes_columns = ['className', 'TP', 'FP', 'npos', 'Recall', 'Precision', 'AP']
image_columns = ['SRC_ID', 'DST_ID', "dataset_name", "name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]
dataset_and_project_columns = ["name", "TP", "FP", 'NPOS', "Precision", "Recall", "mAP"]

