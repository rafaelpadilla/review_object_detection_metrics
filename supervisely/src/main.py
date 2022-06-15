import supervisely as sly
import globals as g
import ui


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id
    })

    data = {}
    state = {}

    # from supervisely.src.ui.test import test
    # test.init_demo_sample(data, state)

    # init data for UI widgets
    ui.init(data, state)
    g.my_app.compile_template(g.root_source_dir)
    g.my_app.run(data=data, state=state)


# @TODO: GTteamId - gtProjectId - camel case
if __name__ == "__main__":
    sly.main_wrapper("main", main)
