import metralign
from drift_sense.localizer import LocalizationConfig, Prediction, localize


def test_metralign_public_api_preserves_compatibility_namespace() -> None:
    assert metralign.LocalizationConfig is LocalizationConfig
    assert metralign.Prediction is Prediction
    assert metralign.localize is localize
    assert metralign.__version__ == "0.2.0"
