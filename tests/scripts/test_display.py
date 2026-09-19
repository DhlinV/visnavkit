import pytest
from lightning.pytorch.callbacks import ModelSummary, RichModelSummary, RichProgressBar, TQDMProgressBar

from visnavkit.utils.display import LineProgressBar, StageSummary, display_callbacks


@pytest.mark.parametrize(
    ("bar", "summary", "types", "kwargs"),
    [
        ("tqdm", "lightning", (TQDMProgressBar, ModelSummary), {}),
        ("lines", "stages", (LineProgressBar, StageSummary), {"enable_model_summary": False}),
        ("rich", "rich", (RichProgressBar, RichModelSummary), {}),
        ("none", "none", (), {"enable_progress_bar": False, "enable_model_summary": False}),
    ],
)
def test_display_styles_pick_their_callbacks(bar, summary, types, kwargs):
    if bar == "rich":
        pytest.importorskip("rich")
    callbacks, extra = display_callbacks({"progress_bar": bar, "model_summary": summary})
    assert tuple(type(c) for c in callbacks) == types and extra == kwargs


def test_trainer_kwargs_that_disable_the_display_win():
    callbacks, extra = display_callbacks({"progress_bar": "tqdm"}, {"enable_progress_bar": False})
    assert not any(isinstance(c, TQDMProgressBar) for c in callbacks) and "enable_progress_bar" not in extra
    with pytest.raises(ValueError, match="progress_bar"):
        display_callbacks({"progress_bar": "fancy"})
