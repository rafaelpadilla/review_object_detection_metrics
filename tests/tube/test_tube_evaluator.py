import os

from src.evaluators.tube_evaluator import TubeEvaluator


# TODO: More tests!!!
def test_tube_eval():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    example_anno = os.path.join(this_dir, 'example_anno.json')
    example_preds = os.path.join(this_dir, 'example_preds.json')

    tube_evaluator = TubeEvaluator(example_anno, example_preds)
    res, mAP = tube_evaluator.evaluate(thr=0.5)

    assert mAP == 1.0
