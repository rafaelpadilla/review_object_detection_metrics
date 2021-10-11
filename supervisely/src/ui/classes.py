import supervisely_lib as sly
import globals as g
import confusion_matrix
import datasets


classes_table = []


def _process_items(collection1, collection2, diff_msg="Automatic conversion to rectangle format"):
    items1 = {item.name: 1 for item in collection1}
    items2 = {item.name: 1 for item in collection2}
    names = items1.keys() | items2.keys()
    mutual = items1.keys() & items2.keys()
    diff1 = items1.keys() - mutual
    diff2 = items2.keys() - mutual

    match = []
    differ = []
    missed = []

    def set_info(d, index, meta):
        d[f"name{index}"] = meta.name
        d[f"color{index}"] = sly.color.rgb2hex(meta.color)
        if type(meta) is sly.ObjClass:
            d[f"shape{index}"] = meta.geometry_type.geometry_name()
            d[f"shapeIcon{index}"] = "zmdi zmdi-shape"
        else:
            meta: sly.TagMeta
            d[f"shape{index}"] = meta.value_type
            d[f"shapeIcon{index}"] = "zmdi zmdi-label"

    for name in names:
        compare = {}
        meta1 = collection1.get(name)
        if meta1 is not None:
            set_info(compare, 1, meta1)
        meta2 = collection2.get(name)
        if meta2 is not None:
            set_info(compare, 2, meta2)

        compare["infoMessage"] = "Match"
        compare["infoColor"] = "green"
        if name in mutual:
            flag = True
            if type(meta1) is sly.ObjClass and meta1.geometry_type != meta2.geometry_type:
                flag = False
            if type(meta1) is sly.TagMeta:
                meta1: sly.TagMeta
                meta2: sly.TagMeta
                if meta1.value_type != meta2.value_type:
                    flag = False
                if meta1.value_type == sly.TagValueType.ONEOF_STRING:
                    if set(meta1.possible_values) != set(meta2.possible_values):
                        diff_msg = "Type OneOf: conflict of possible values"
                    flag = False

            if flag is False:
                compare["infoMessage"] = diff_msg
                compare["infoColor"] = "red"
                compare["infoIcon"] = ["zmdi zmdi-flag"],
                differ.append(compare)
            else:
                compare["infoIcon"] = ["zmdi zmdi-check"],
                match.append(compare)
        else:
            if name in diff1:
                compare["infoMessage"] = "Not found in PRED Project"
                compare["infoIcon"] = ["zmdi zmdi-alert-circle-o", "zmdi zmdi-long-arrow-right"]
                compare["iconPosition"] = "right"
            else:
                compare["infoMessage"] = "Not found in GT Project"
                compare["infoIcon"] = ["zmdi zmdi-long-arrow-left", "zmdi zmdi-alert-circle-o"]
            compare["infoColor"] = "#FFBF00"
            missed.append(compare)

    table = []
    if match:
        match.sort(key=lambda x: x['name1'])
    table.extend(match)
    if differ:
        differ.sort(key=lambda x: x['name1'])
    table.extend(differ)
    table.extend(missed)

    return table


def init(data, state):
    global classes_table
    # try:
    #     if g.aggregated_meta is None:
    #         g.generate_meta()
    #     classes_table = _process_items(g.gt_meta.obj_classes, g.pred_meta.obj_classes)
    #     data["classesTable"] = classes_table
    # except:
    #     pass
    data["classesTable"] = None
    state["selectedClasses"] = []
    state['collapsed3'] = True
    state['disabled3'] = True
    state['disabled3Btn'] = False
    state['done3'] = False


def restart(data, state):
    state['collapsed3'] = False
    state['disabled3'] = False
    state['disabled3Btn'] = False
    state['done3'] = False
    state["selectedClasses"] = []


@g.my_app.callback("get_classes")
@sly.timeit
def get_classes(api: sly.Api, task_id, context, state, app_logger):
    global classes_table
    classes_table = _process_items(g.gt_meta.obj_classes, g.pred_meta.obj_classes)
    fields = [
        # {"field": "state.done3", "payload": True},
        {"field": "data.classesTable", "payload": classes_table},
    ]
    api.app.set_fields(task_id, fields)


@g.my_app.callback("set_classes")
@sly.timeit
def set_classes(api: sly.Api, task_id, context, state, app_logger):
    total_img_num = 0
    for k, v in datasets.image_dict['gt_images'].items():
        total_img_num += len(v)

    fields = []
    # for i in range(1, 6):
    #     collapsed = True if i != 4 else False
    #     disabled = True if i not in [2, 3, 4] else False
    #     done = True if i < 4 else False
    #     fields.append({"field": f"state.collapsed{i}", "payload": collapsed})
    #     fields.append({"field": f"state.disabled{i}", "payload": disabled})
    #     fields.append({"field": f"state.done{i}", "payload": done})

    extra_fields = [
        {"field": "state.done3", "payload": True},

        {"field": "state.collapsed4", "payload": False},
        {"field": "state.disabled4", "payload": False},

        {"field": "state.activeStep", "payload": 4},
        {"field": "state.GlobalShowSettings", "payload": True},

        {"field": "state.GlobalShowMetrics", "payload": False},
        {"field": "data.totalImagesCount", "payload": total_img_num},
    ]
    fields.extend(extra_fields)

    api.app.set_fields(task_id, fields)
    confusion_matrix.reset_cm_state_to_default(api, task_id)

