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

# @TODO: show-input в слайдеры и сделать их по размеру 450px
# @TODO: Compare Datasets descr: += Only matched images are used in ....
# @TODO: BUTTON_NAAME -> Calculate metrics
# @TODO: обираем список selected classes
# @TODO: подключить виджет галереи из SDK sly.app.widgets.CompareGallery

# @TODO: Images for the selected cel in confusion matrix: bird (actual) <-> "cat" (predicted)
# @TODO: DESCRIPTION 2 "bird" objects are detected as "dogs"
# @TODO: DESCRIPTION 2 "bird" objects are not detected (bird <-> None)
# @TODO: DESCRIPTION Model predicted 77 "cat" objects that are not in GT (None <-> Bird)
# @TODO: Umar - скрол горизонтальный для confusion matrix - на стороне виджета?
# @TODO: Umar - настройка толщины прямоугольников и полигонов в grid gallery
# delete Image Statistic Table.
# Image_mAP -> Metrics pre images 
# Evaluate -> Preview
# Extra Class Info -> убрать карточку
# dog FP 10 vs 7 BUG
# Extra Class Info -> cart to field, remove table
# Sly Field - добавить перед таблицей для выбранного класса

# @TODO: check requirements - two files instead of one
# @TODO: disable class selection for conflicts
# @TODO: clear global unused requirements
# @TODO: remove zero cell clickability (Umar) - wip
# @TODO: check plt2bb in utils.py for bugs
if __name__ == "__main__":
    sly.main_wrapper("main", main)
