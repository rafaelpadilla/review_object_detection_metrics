import json


def parse_cvat(file_path):
    """ Parse a cvat file, returning a list with annotations.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    list
        List of dictionaries with annotations in the cvat file.
    """

    with open(file_path, "r") as f:
        data = json.load(f)

    shape_keys = [
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
    ]
    shapes = [
        dict(
            label=s["label"],
            points=s["points"],
            shape_type=s.get("shape_type", "polygon"),
            flags=s.get("flags", {}),
            group_id=s.get("group_id"),
            other_data={k: v
                        for k, v in s.items() if k not in shape_keys},
        ) for s in data["shapes"]
    ]
    return shapes
