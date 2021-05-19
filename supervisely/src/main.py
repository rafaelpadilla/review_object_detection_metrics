import supervisely_lib as sly
import globals as g
import ui
import os


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        # "modal.state.slyProjectId": project_id,  # @TODO: log more input envs
    })
    try:
        os.remove('db')
    except:
        pass

    data = {}
    state = {}
    g.my_app.compile_template(g.root_source_dir)

    # init data for UI widgets
    ui.init(data, state)
    g.my_app.run(data=data, state=state)


# @TODO: check requirements - two files instead of one
# @TODO: disable class selection for conflicts
# @TODO: clear global unused requirements
# @TODO: remove zero cell clickability (Umar)
if __name__ == "__main__":
    sly.main_wrapper("main", main)
