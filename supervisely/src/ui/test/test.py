# demo data
# ??? sort with empty cells
def entity(id_, class_, color_):
    data = {
        "id": id_,
        "class": class_,
        "color": color_
    }
    return data


def template(gt, pred, iou, confidence):
    data = {
        "gt": gt,
        "pred": pred,
        "iou": iou,
        "confidence": confidence
    }
    return data


def init_demo_sample(data, state):
    data["demoData"] = [
        {
            "gt": {
                "class": "car",
                "color": "#3F00FF",
                "id": 777
            },
            "iou": 0.7,
            "conf": 0.9999,
            "pred": {
                "class": "person",
                "color": "#FBAD00",
                "id": 888
            },
            "id_pair": [777, 888]
        },
        {
            "gt": {
                "class": "dog",
                "color": "#3F00FF",
                "id": 111
            },
            "iou": 0,
            "conf": 0,
            "id_pair": [111, None]
        },
        {
            "iou": 0,
            "conf": 0.1,
            "pred": {
                "class": "cat",
                "color": "#FBAD00",
                "id": 222
            },
            "id_pair": [None, 222]
        },

        # },
        # {
        #     "date": '2016-05-02',
        #     "name": 'Tom',
        #     "address": 'No. 189, Grove St, Los Angeles'
        # },
        # {
        #     "date": '2016-05-04',
        #     "name": 'Tom',
        #     "address": 'No. 189, Grove St, Los Angeles'
        # },
        # {
        #     "date": '2016-05-01',
        #     "name": 'Tom',
        #     "address": 'No. 189, Grove St, Los Angeles'
        # }
    ]
