python cli.py \
--anno_gt ./toyexample/gts_vocpascal_format/ \
--anno_det ./toyexample/dets_classid_abs_xywh \
--gtformat voc \
--detformat xywh \
--detcoord abs \
--metric coco \
--names ./toyexample/voc.names \
-sp ./cli_test \
-pr