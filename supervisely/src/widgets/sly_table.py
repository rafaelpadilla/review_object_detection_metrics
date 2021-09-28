from typing import Union
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.api.api import Api
from supervisely_lib.annotation.annotation import Annotation


class SlyTable:
    def __init__(self, api: Api, task_id, v_model, columns):
        self._api = api
        self._task_id = task_id
        self._v_model = v_model
        self._columns = columns
        self._data = []

    def add_rows(self, rows):
        self.data.append(rows)

    def set_data(self, data):
        self._data = data

    def update(self):
        table_json = self.to_json()
        self._api.task.set_field(self._task_id, self._v_model, table_json)

    def to_json(self):
        return {
            "columns": self._columns,
            "data": self._data
        }
