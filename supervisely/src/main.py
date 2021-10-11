import supervisely_lib as sly
import globals as g
import ui
import os


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id
    })
    # try:
    #     os.remove('db')
    # except:
    #     pass

    data = {}
    state = {}

    from supervisely.src.ui.test import test
    test.init_demo_sample(data, state)

    # init data for UI widgets
    ui.init(data, state)
    g.my_app.compile_template(g.root_source_dir)
    g.my_app.run(data=data, state=state)

    #@TODO: GTteamId - gtProjectId - нужно сделать именования везде однородные, начинаем с маленькой буквы
    #@TODO:


if __name__ == "__main__":
    sly.main_wrapper("main", main)
