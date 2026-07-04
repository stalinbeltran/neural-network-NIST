"""Tests del registro de resultados multi-métrica."""
from nnist.evaluation import RunResult


def test_run_result_serializes():
    r = RunResult(
        run_id="mlp_full_test",
        model_name="mlp",
        strategy="full",
        input_shape=(1, 28, 28),
        num_classes=10,
        params_total=101770,
        params_trainable=101770,
        accuracy=0.97,
    )
    d = r.as_dict()
    assert d["accuracy"] == 0.97
    assert d["params_total"] == 101770
