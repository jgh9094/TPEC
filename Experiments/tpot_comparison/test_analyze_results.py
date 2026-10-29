"""Tests for the one-row-per-run TPOT analysis export."""

import json
import tempfile
import unittest
from pathlib import Path

from Experiments.tpot_comparison.analyze_results import (
    RUN_OUTPUT_COLUMNS,
    build_run_output,
    collect_results,
    condition_label,
)


class RunOutputTests(unittest.TestCase):
    def test_requested_run_csv_includes_100_and_500_evaluation_graph_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_root = Path(temporary_directory)
            run_directory = (
                results_root
                / "Pop25"
                / "Graph"
                / "Task_HOSP_READM_90"
                / "Seed_0"
            )
            run_directory.mkdir(parents=True)
            (run_directory / "best_results.json").write_text(
                json.dumps(
                    {
                        "metric": "roc_auc",
                        "train_auroc": 0.8683582673874204,
                        "validation_auroc": 0.7146209782103702,
                        "test_auroc": 0.6998322785391419,
                        "complexity": 127877,
                        "num_evaluated_individuals": 500,
                        "fit_time_minutes": 104.41221197843552,
                    }
                ),
                encoding="utf-8",
            )

            run_directory_100 = (
                results_root
                / "Pop25"
                / "Graph_100"
                / "Task_HOSP_READM_90"
                / "Seed_0"
            )
            run_directory_100.mkdir(parents=True)
            (run_directory_100 / "best_results.json").write_text(
                json.dumps(
                    {
                        "metric": "roc_auc",
                        "train_auroc": 0.81,
                        "validation_auroc": 0.70,
                        "test_auroc": 0.69,
                        "complexity": 200,
                        "num_evaluated_individuals": 100,
                        "fit_time_minutes": 25.0,
                    }
                ),
                encoding="utf-8",
            )
            (run_directory_100 / "archive.json").write_text(
                json.dumps(
                    [
                        {
                            "individual_id": "0",
                            "roc_auc_score": 0.70,
                            "complexity_scorer": 200,
                            "Generation": 0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (run_directory / "configuration.json").write_text(
                json.dumps(
                    {
                        "objectives": {
                            "performance_history_column": "roc_auc_score",
                            "complexity": "complexity_scorer",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_directory / "archive.json").write_text(
                json.dumps(
                    [
                        {
                            "individual_id": "0",
                            "roc_auc_score": 0.65,
                            "complexity_scorer": 10,
                            "Generation": 0,
                        },
                        {
                            "individual_id": "67",
                            "roc_auc_score": 0.7146209782103702,
                            "complexity_scorer": 127877,
                            "Generation": 2,
                        },
                        {
                            "individual_id": "99",
                            "roc_auc_score": 0.70,
                            "complexity_scorer": 20,
                            "Generation": 3,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            results = collect_results(
                results_root=results_root,
                pop_size=25,
                search_spaces=["Graph", "Graph_100"],
                tasks=["HOSP_READM_90"],
                seeds=[0],
            )
            output = build_run_output(results)

            self.assertEqual(tuple(output.columns), RUN_OUTPUT_COLUMNS)
            self.assertEqual(len(output), 2)
            self.assertEqual(set(output["evals"]), {100, 500})
            self.assertEqual(set(output["condition"]), {"tpot_g"})

            row = output.loc[output["evals"].eq(500)].iloc[0]
            self.assertEqual(row["task"], "HOSP_READM_90")
            self.assertEqual(row["seed"], 0)
            self.assertEqual(row["best_eval"], 68)
            self.assertEqual(row["best_generation"], 3)
            self.assertAlmostEqual(row["train_auroc"], 0.8683582673874204)
            self.assertAlmostEqual(
                row["validation_auroc"], 0.7146209782103702
            )
            self.assertAlmostEqual(row["test_auroc"], 0.6998322785391419)
            self.assertEqual(row["complexity"], 127877)
            self.assertAlmostEqual(
                row["total_runtime_minutes"], 104.41221197843552
            )

            row_100 = output.loc[output["evals"].eq(100)].iloc[0]
            self.assertEqual(row_100["best_generation"], 1)

    def test_condition_labels(self):
        self.assertEqual(condition_label("Constrained"), "tpot_c")
        self.assertEqual(condition_label("Constrained_100"), "tpot_c")
        self.assertEqual(condition_label("Graph"), "tpot_g")
        self.assertEqual(condition_label("Graph_100"), "tpot_g")


if __name__ == "__main__":
    unittest.main()
