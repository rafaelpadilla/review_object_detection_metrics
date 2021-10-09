import supervisely_lib as sly
import globals as g
import ui
import os


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id
    })
    try:
        os.remove('db')
    except:
        pass

    data = {}
    state = {}

    # demo data
    # ??? sort with empty cells
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
        },
        {
            "gt": {
                "class": "dog",
                "color": "#3F00FF",
                "id": 111
            },
            "iou": 0,
            "conf": 0,
        },
        {
            "iou": 0,
            "conf": 0.1,
            "pred": {
                "class": "cat",
                "color": "#FBAD00",
                "id": 222
            },
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

    # init data for UI widgets
    ui.init(data, state)
    g.my_app.compile_template(g.root_source_dir)
    g.my_app.run(data=data, state=state)

    # @TODO: DESCRIPTION 2 "bird" objects are not detected (bird <-> None) +
    # @TODO: DESCRIPTION Model predicted 77 "cat" objects that are not in GT (None <-> Bird) +
    # легенда для таблицы с изображениями +
    # dog FP 10 vs 7 BUG ?
    # @TODO: clear global unused requirements
    # @TODO: check plt2bb in utils.py for bugs

    # @TODO: Umar - настройка толщины прямоугольников и полигонов в grid gallery - связаться с Антоном

if __name__ == "__main__":
    sly.main_wrapper("main", main)
