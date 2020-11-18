import pytest
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType


def test_input():
    with pytest.raises(IOError):
        BoundingBox("image", bb_type=BBType.DETECTED, class_confidence=None)
