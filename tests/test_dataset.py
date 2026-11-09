import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refutebench import dataset, detect, evaluate


def test_loads_and_validates():
    items = dataset.load()
    assert len(items) >= 20
    assert any(i.has_defect for i in items)
    assert any(not i.has_defect for i in items), "need clean controls for precision"


def test_labels_are_consistent():
    for i in dataset.load():
        assert i.has_defect == (i.defect_class != "none"), i.id


def test_clean_controls_are_a_real_fraction():
    items = dataset.load()
    clean = sum(not i.has_defect for i in items)
    assert clean / len(items) >= 0.15, "too few negatives to measure precision"


def test_parser_handles_garbage_as_clean_not_skip():
    p = detect.parse("X", "the model rambled and never answered")
    assert p.has_defect is False and p.defect_class == "none"


def test_parser_reads_a_well_formed_answer():
    p = detect.parse("X", "VERDICT: DEFECT\nCLASS: look_ahead\nWHY: uses future data.")
    assert p.has_defect and p.defect_class == "look_ahead" and "future" in p.why


def test_scoring_arithmetic():
    items = dataset.load()
    preds = [detect.Prediction(i.id, i.has_defect, i.defect_class, "", "") for i in items]
    s = evaluate.score(items, preds, "oracle")
    assert s.accuracy == 1.0 and s.fp == 0 and s.fn == 0
    inv = [detect.Prediction(i.id, not i.has_defect, "", "", "") for i in items]
    s2 = evaluate.score(items, inv, "anti-oracle")
    assert s2.accuracy == 0.0
