"""Regression tests for the operator-matched TPOT search spaces."""

import json
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tpot
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from Source.Base.model_param_space import DataContext

from Experiments.tpot_comparison.param_space_conversion import (
    generate_tpot_graph_search_space,
    generate_tpot_search_space,
)
from Experiments.tpot_comparison.tpot_wrapper import (
    DEFAULT_MAX_EVAL_TIME_MINS,
    classification_metrics,
    cross_validated_classification_metrics,
    cross_validated_objectives,
    load_spine_opioid_data,
    make_preprocessor,
    prepare_cv_folds,
)


def _value_signature(value):
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _parser_signature(parser):
    if isinstance(parser, partial):
        return (
            parser.func.__module__,
            parser.func.__qualname__,
            tuple(_value_signature(value) for value in parser.args),
            tuple(
                sorted(
                    (key, _value_signature(value))
                    for key, value in parser.keywords.items()
                )
            ),
        )
    return (parser.__module__, parser.__qualname__)


def _node_signature(node):
    space = node.space
    serialized_space = (
        json.dumps(space.to_serialized_dict(), sort_keys=True)
        if hasattr(space, "to_serialized_dict")
        else json.dumps(space, sort_keys=True)
    )
    return (
        node.method.__module__,
        node.method.__qualname__,
        serialized_space,
        _parser_signature(node.hyperparameter_parser),
    )


class OperatorMatchedSearchSpaceTests(unittest.TestCase):
    def test_spinal_surgery_numpy_dataset_loader(self):
        dataset_directory = (
            Path(__file__).resolve().parents[2] / "Data-Tools" / "spinal_surgery"
        )
        X, y, numerical_columns, categorical_columns = load_spine_opioid_data(
            dataset_directory,
            "LOS_extended",
        )

        self.assertEqual(X.shape, (6365, 70))
        self.assertEqual(y.shape, (6365,))
        self.assertIn("PLATELET_COUNT", X.columns)
        self.assertIn("WBC_COUNT", X.columns)
        self.assertIn("RBC_COUNT", X.columns)
        self.assertNotIn("PAT_ID", X.columns)
        self.assertTrue(set(numerical_columns).issubset(X.columns))
        self.assertEqual(categorical_columns, [])

    def assert_operator_spaces_match(self, classification):
        data_context = DataContext(n_samples=80, n_features=12, n_classes=2)
        kwargs = {
            "data_context": data_context,
            "seed": 7,
            "classification": classification,
            "scale_columns": (0, 1, 2),
        }
        constrained = generate_tpot_search_space(**kwargs)
        graph = generate_tpot_graph_search_space(**kwargs, n_folds=5)

        constrained_nodes = {
            _node_signature(node)
            for stage in constrained.search_spaces
            for node in stage.search_spaces
        }
        graph_inner_nodes = {
            _node_signature(node) for node in graph.inner_search_space.search_spaces
        }
        graph_root_nodes = {
            _node_signature(node) for node in graph.root_search_space.search_spaces
        }
        constrained_predictors = {
            _node_signature(node)
            for node in constrained.search_spaces[-1].search_spaces
        }

        self.assertEqual(constrained_nodes, graph_inner_nodes)
        self.assertEqual(constrained_predictors, graph_root_nodes)
        self.assertEqual(len(constrained_predictors), 6)
        self.assertEqual(len(constrained_nodes), 24)
        self.assertFalse(
            any(signature[1] == "PolynomialFeatures" for signature in constrained_nodes)
        )

        # Both search spaces must generate exportable sklearn-compatible pipelines.
        for rng_seed in range(10):
            self.assertIsNotNone(constrained.generate(rng=rng_seed).export_pipeline())
            self.assertIsNotNone(graph.generate(rng=rng_seed).export_pipeline())

    def test_classification_operator_spaces_match(self):
        self.assert_operator_spaces_match(classification=True)

    def test_regression_operator_spaces_match(self):
        self.assert_operator_spaces_match(classification=False)

    def test_custom_objective_uses_fold_local_preprocessing(self):
        X = pd.DataFrame(
            {
                "numeric": np.arange(9, dtype=float),
                "category": ["validation_only"] * 3 + ["common"] * 6,
            }
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
        cv_splits = list(KFold(n_splits=3, shuffle=False).split(X, y))
        fold_data = prepare_cv_folds(
            X_train=X,
            y_train=y,
            cv_splits=cv_splits,
            numerical_columns=("numeric",),
            categorical_columns=("category",),
        )

        # The first fold's training partition contains only "common". Its one-hot
        # output therefore has one numeric and one categorical column, proving that
        # the validation-only category was not learned before validation.
        self.assertEqual(fold_data[0][0].shape[1], 2)
        self.assertEqual(fold_data[0][1].shape[1], 2)

        candidate = LogisticRegression(max_iter=100)
        objectives = cross_validated_objectives(
            candidate,
            fold_data=fold_data,
            classification=True,
            classes=np.array([0, 1]),
        )
        self.assertEqual(objectives.shape, (2,))
        self.assertTrue(np.isfinite(objectives).all())
        expected_complexities = []
        for X_fold_train, _, y_fold_train, _ in fold_data:
            fitted_candidate = clone(candidate).fit(X_fold_train, y_fold_train)
            expected_complexities.append(
                tpot.objectives.complexity_scorer(fitted_candidate)
            )
        self.assertAlmostEqual(objectives[1], np.mean(expected_complexities))

        validation_metrics = cross_validated_classification_metrics(
            candidate,
            fold_data=fold_data,
            classes=np.array([0, 1]),
        )
        self.assertEqual(
            set(validation_metrics),
            {"auroc", "auprc", "balanced_accuracy"},
        )
        self.assertTrue(
            all(0.0 <= value <= 1.0 for value in validation_metrics.values())
        )

    def test_final_preprocessor_stays_outside_candidate_pipeline(self):
        X = pd.DataFrame(
            {
                "numeric": np.arange(9, dtype=float),
                "category": ["validation_only"] * 3 + ["common"] * 6,
            }
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
        preprocessor = make_preprocessor(
            numerical_columns=("numeric",),
            categorical_columns=("category",),
        )
        X_transformed = preprocessor.fit_transform(X)
        candidate = LogisticRegression(max_iter=100)
        candidate.fit(X_transformed, y)

        learned_categories = (
            preprocessor.named_transformers_["cat"].categories_[0]
        )
        self.assertEqual(
            list(learned_categories),
            ["common", "validation_only"],
        )
        self.assertFalse(hasattr(candidate, "named_steps"))
        self.assertEqual(len(candidate.predict(X_transformed)), len(y))
        metrics = classification_metrics(
            candidate, X_transformed, y, np.array([0, 1])
        )
        self.assertEqual(set(metrics), {"auroc", "auprc", "balanced_accuracy"})
        self.assertTrue(all(0.0 <= value <= 1.0 for value in metrics.values()))

    def test_default_pipeline_evaluation_limit_is_five_minutes(self):
        self.assertEqual(DEFAULT_MAX_EVAL_TIME_MINS, 5.0)


if __name__ == "__main__":
    unittest.main()
