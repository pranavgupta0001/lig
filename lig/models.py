"""Model selection, training, and feature-importance utilities."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from .preprocessing import SmilesEncoder


class ModelFactory:
    """Create supported predictors from a stable model name."""

    @staticmethod
    def create(model: str, random_state: int = 42):
        name = model.lower().replace("-", "_").replace(" ", "_")
        try:
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
            from sklearn.neural_network import MLPRegressor
        except ImportError as exc:
            raise ImportError(
                "Install the 'ml' extra to train LIG models: pip install -e '.[ml]'"
            ) from exc

        if name in {"random_forest", "rf"}:
            return RandomForestRegressor(n_estimators=300, random_state=random_state)
        if name in {"gradient_boosting", "gradient_boost", "gbr"}:
            return GradientBoostingRegressor(random_state=random_state)
        if name in {"neural_network", "mlp", "nn"}:
            return MLPRegressor(
                hidden_layer_sizes=(128, 64),
                early_stopping=True,
                max_iter=1000,
                random_state=random_state,
            )
        if name in {"xgboost", "xgb"}:
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "Install XGBoost to use the xgboost model"
                ) from exc
            return XGBRegressor(
                n_estimators=300,
                random_state=random_state,
                objective="reg:squarederror",
            )
        raise ValueError(
            "model must be random_forest, gradient_boosting, neural_network, or xgboost"
        )


def train_and_evaluate(
    smiles: Iterable[str],
    targets,
    representation: str = "fingerprint",
    model: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Encode SMILES and train/evaluate a single-target LIG predictor."""
    try:
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "Install the 'ml' extra to train LIG models: pip install -e '.[ml]'"
        ) from exc

    smiles = list(smiles)
    targets = np.asarray(list(targets), dtype=float)
    if len(smiles) == 0:
        raise ValueError("smiles must contain at least one value")
    if len(smiles) != len(targets):
        raise ValueError("smiles and targets must have the same length")
    if len(smiles) < 2:
        raise ValueError("At least two samples are required")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    features = np.asarray(
        SmilesEncoder(representation=representation).fit_transform(smiles),
        dtype=float,
    )
    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
    )
    predictor = ModelFactory.create(model, random_state=random_state)
    predictor.fit(train_features, train_targets)
    predictions = predictor.predict(test_features)
    report = OrderedDict(
        [
            ("model", model.lower().replace("-", "_").replace(" ", "_")),
            ("representation", representation),
            ("n_samples", len(smiles)),
            ("n_train", len(train_targets)),
            ("n_test", len(test_targets)),
            ("mae", float(mean_absolute_error(test_targets, predictions))),
            ("rmse", float(mean_squared_error(test_targets, predictions) ** 0.5)),
            ("r2", float(r2_score(test_targets, predictions))),
        ]
    )
    return predictor, dict(report)


def feature_importance(predictor) -> list[float]:
    """Return model feature importances, or raise for a model without them."""
    if not hasattr(predictor, "feature_importances_"):
        raise AttributeError("predictor does not expose feature_importances_")
    return [float(value) for value in predictor.feature_importances_]
