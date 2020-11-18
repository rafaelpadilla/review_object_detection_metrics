from src.enumerators import BBFormat


def test_BBFormat():
    assert BBFormat.XYWH._value_ == 1
    assert BBFormat.XYX2Y2._value_ == 2
    assert BBFormat.PASCAL_XML._value_ == 3
    assert BBFormat.YOLO._value_ == 4
