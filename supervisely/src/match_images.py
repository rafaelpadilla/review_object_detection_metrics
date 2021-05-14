import supervisely_lib as sly
import globals as g


def match(gt_project_id, pred_project_id):
    gt_datasets = g.api.dataset.get_list(gt_project_id)
    pred_datasets = g.api.dataset.get_list(pred_project_id)

    pred_datasets_by_name = {item.name: item for item in pred_datasets}
    matched_datasets = []
    for dataset_info in gt_datasets:
        if dataset_info.name not in pred_datasets_by_name:
            continue
        matched_datasets.append((dataset_info, pred_datasets_by_name[dataset_info.name]))

    for (gt_dataset, pred_dataset) in matched_datasets:
        pass
