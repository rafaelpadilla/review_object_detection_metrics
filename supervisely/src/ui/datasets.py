import supervisely_lib as sly
import globals as g

global RESULTS, RESULTS_DATA, image_dict, total_img_num
RESULTS = None
RESULTS_DATA = None
image_dict = {'gt_images': {}, 'pred_images': {}}


def process_items(ds_info1, collection1, ds_info2, collection2):
    global RESULTS, RESULTS_DATA
    ds_names = ds_info1.keys() | ds_info2.keys()

    results = []
    results_data = []
    for idx, name in enumerate(ds_names):
        compare = {"dsIndex": idx}
        images1 = collection1.get(name, [])
        images2 = collection2.get(name, [])
        if len(images1) == 0:
            compare["message"] = ["unmatched (in GT project)"]
            compare["icon"] = [["zmdi zmdi-long-arrow-left", "zmdi zmdi-alert-circle-o"]]

            compare["color"] = ["#F39C12"]
            compare["numbers"] = [-1]
            compare["left"] = {"name": ""}
            compare["right"] = {"name": name, "count": len(images2)}
            results_data.append([images2])
        elif len(images2) == 0:
            compare["message"] = ["unmatched (in PRED project)"]
            compare["icon"] = [["zmdi zmdi-alert-circle-o", "zmdi zmdi-long-arrow-right"]]
            compare["color"] = ["#F39C12"]
            compare["numbers"] = [-1]
            compare["left"] = {"name": name, "count": len(images1)}
            compare["right"] = {"name": ""}
            results_data.append([images1])
        else:
            img_dict1 = {img_info.name: img_info for img_info in images1}
            img_dict2 = {img_info.name: img_info for img_info in images2}

            matched = []
            diff = []  # same names but different hashes or image sizes
            same_names = img_dict1.keys() & img_dict2.keys()
            for img_name in same_names:
                dest = matched if img_dict1[img_name].hash == img_dict2[img_name].hash else diff
                dest.append([img_dict1[img_name], img_dict2[img_name]]) # extend

            uniq1 = [img_dict1[name] for name in img_dict1.keys() - same_names]
            uniq2 = [img_dict2[name] for name in img_dict2.keys() - same_names]

            compare["message"] = ["matched", "conflicts", "unique (left)", "unique (right)"]
            compare["icon"] = [["zmdi zmdi-check"], ["zmdi zmdi-close"], ["zmdi zmdi-plus-circle-o"],
                               ["zmdi zmdi-plus-circle-o"]]
            compare["color"] = ["green", "red", "#20a0ff", "#20a0ff"]
            compare["numbers"] = [len(matched), len(diff), len(uniq1), len(uniq2)]
            compare["left"] = {"name": name, "count": len(images1)}
            compare["right"] = {"name": name, "count": len(images2)}
            results_data.append([matched, diff, uniq1, uniq2])

        results.append(compare)

    RESULTS = results
    RESULTS_DATA = results_data
    return results


def _get_all_images(api: sly.Api, project):
    ds_info = {}
    ds_images = {}
    ws_to_team = {}
    for dataset in api.dataset.get_list(project.id):
        ds_info[dataset.name] = dataset
        images = api.image.get_list(dataset.id)
        modified_images = []
        for image_info in images:
            if project.workspace_id not in ws_to_team:
                ws_to_team[project.workspace_id] = api.workspace.get_info_by_id(project.workspace_id).team_id
            meta = {
                "team_id": ws_to_team[project.workspace_id],
                "workspace_id": project.workspace_id,
                "project_id": project.id,
                "project_name": project.name,
                "dataset_name": dataset.name,
                "meta": image_info.meta
            }
            image_info = image_info._replace(meta=meta)
            modified_images.append(image_info)
        ds_images[dataset.name] = modified_images
    return ds_info, ds_images,


def init(data, state, reconstruct=False):
    state['GlobalDatasetsCollapsed'] = True
    state['GlobalDatasetsDisabled'] = True
    state['doneDatasets'] = False
    state['DatasetsInProgress'] = False
    state['datasetReconstruct'] = False


def restart(data, state):
    state['doneDatasets'] = False
    state['GlobalActiveStep'] = 2
    state['GlobalDatasetsCollapsed'] = False
    state['GlobalDatasetsDisabled'] = False
    state['GlobalClassesCollapsed'] = True
    state['GlobalClassesDisabled'] = True
    state['GlobalSettingsCollapsed'] = True
    state['GlobalSettingsDisabled'] = True
    state['GlobalMetricsCollapsed'] = True
    state['GlobalMetricsDisabled'] = True


@g.my_app.callback("get_datasets_statistic")
@sly.timeit
def get_datasets_statistic(api: sly.Api, task_id, context, state, app_logger):
    global image_dict, total_img_num

    g.api.app.set_field(g.task_id, "state.DatasetsInProgress", True)

    g.gt_project_info = api.project.get_info_by_id(state['gtProjectId'], raise_error=True)
    g._gt_meta_ = api.project.get_meta(state['gtProjectId'])
    g.gt_meta = sly.ProjectMeta.from_json(g._gt_meta_)

    g.pred_project_info = api.project.get_info_by_id(state['predProjectId'], raise_error=True)
    g._pred_meta_ = api.project.get_meta(state['predProjectId'])
    g.pred_meta = sly.ProjectMeta.from_json(g._pred_meta_)
    g.generate_meta()

    ds_info1, ds_images1 = _get_all_images(g.api, g.gt_project_info)
    ds_info2, ds_images2 = _get_all_images(g.api, g.pred_project_info)
    result = process_items(ds_info1, ds_images1, ds_info2, ds_images2)
    intersected_keys = list(set(list(ds_images1)) & set(list(ds_images2)))
    # image_dict = {'gt_images': {}, 'pred_images': {}}

    for intersected_key in intersected_keys:
        image_dict['gt_images'][intersected_key] = []
        image_dict['pred_images'][intersected_key] = []

        for gt_element in ds_images1[intersected_key]:
            for pred_element in ds_images2[intersected_key]:
                if gt_element.hash == pred_element.hash and gt_element.name == pred_element.name:
                    image_dict['gt_images'][intersected_key].append(gt_element)
                    image_dict['pred_images'][intersected_key].append(pred_element)

    fields = [
        {"field": "data.table", "payload": result},
        {"field": "state.DatasetsInProgress", "payload": False},
        {"field": "state.doneDatasets", "payload": True},
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("next_step")
@sly.timeit
def next_step(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        # {"field": "state.GlobalDatasetsCollapsed", "payload": True},
        {"field": "state.GlobalDatasetsDisabled", "payload": False}, #@TODO: ?
        {"field": "state.GlobalActiveStep", "payload": 3},
        {"field": "state.doneDatasets", "payload": True},
        {"field": "state.GlobalClassesCollapsed", "payload": False},
        {"field": "state.GlobalClassesDisabled", "payload": False}
    ]
    api.app.set_fields(task_id, fields)
