import os
import supervisely_lib as sly
import globals as g
import ui
import download_data as dd


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        # "modal.state.slyProjectId": project_id,  # @TODO: log more input envs
    })

    data = {}
    state = {}

    g.my_app.compile_template(g.root_source_dir)

    # init data for UI widgets
    ui.init(data, state)

    dd.download(percentage=1, image_dict=ui.datasets.image_dict)
    print(dd.total_image_num)
    for k, v in dd.slice_.items():
        print(k, len(v))

    g.my_app.run(data=data, state=state)


# @TODO: check requirements - two files instead of one
# @TODO: disable class selection for conflicts
# @TODO: clear global unused requirements
if __name__ == "__main__":
    sly.main_wrapper("main", main)
