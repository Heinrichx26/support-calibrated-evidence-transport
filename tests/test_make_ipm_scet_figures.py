from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import pandas as pd

import make_ipm_scet_figures as figures


class IPMSCETFigureTests(unittest.TestCase):
    def test_best_comparison_rows_selects_non_scet_reference_rows(self) -> None:
        summary = pd.DataFrame(
            {
                "dataset": ["scifact", "scifact", "scifact", "fever", "fever", "fever", "hover", "hover", "hover"],
                "retriever": ["bm25", "bge_m3", "scet"] * 3,
                "support_precision": [0.10, 0.20, 0.30, 0.15, 0.25, 0.40, 0.50, 0.60, 0.70],
                "recall_at_5": [0.70, 0.80, 0.79, 0.75, 0.77, 0.78, 0.88, 0.90, 0.91],
            }
        )
        stress = pd.DataFrame(
            {
                "dataset": ["scifact", "scifact", "scifact", "fever", "fever", "fever", "hover", "hover", "hover"],
                "retriever": ["bm25", "bge_m3", "scet"] * 3,
                "stress_support_precision": [0.05, 0.08, 0.12, 0.03, 0.04, 0.09, 0.20, 0.22, 0.30],
            }
        )

        table = figures.best_comparison_rows(summary, stress)

        self.assertEqual(table["label"].tolist(), ["SciFact", "FEVER", "HoVer"])
        self.assertEqual(table["comparison_precision"].tolist(), [0.20, 0.25, 0.60])
        self.assertEqual(table["comparison_stress"].tolist(), [0.08, 0.04, 0.22])
        self.assertAlmostEqual(table.iloc[0]["recall_delta_vs_best"], -0.01)

    def test_export_bundle_keeps_submission_formats(self) -> None:
        self.assertEqual(figures.EXPORT_EXTENSIONS, (".pdf", ".svg", ".png", ".tif"))

    def test_sci_style_uses_editable_vector_text(self) -> None:
        figures.set_style()

        self.assertEqual(plt.rcParams["svg.fonttype"], "none")
        self.assertEqual(plt.rcParams["pdf.fonttype"], 42)
        self.assertIn("serif", plt.rcParams["font.family"])

    def test_finish_axis_keeps_box_spines_visible(self) -> None:
        _, ax = plt.subplots(figsize=(2.0, 1.4))

        figures.finish_axis(ax)

        self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
