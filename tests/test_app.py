"""Run with:  python -m unittest discover -s tests -v"""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import create_app  # noqa: E402
from detector import DetectionError, Params, detect_anomalies, load_numeric_csv  # noqa: E402


def planted_csv(n_normal=500, n_outliers=25) -> tuple[bytes, np.ndarray]:
    """2-D Gaussian blob plus far-away outliers appended at the end; returns (csv, truth)."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, (n_normal, 2))
    outliers = rng.uniform(8, 12, (n_outliers, 2)) * rng.choice([-1, 1], (n_outliers, 2))
    df = pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b"])
    truth = np.r_[np.zeros(n_normal, bool), np.ones(n_outliers, bool)]
    return df.to_csv(index=False).encode(), truth


class DetectorTests(unittest.TestCase):
    def test_finds_planted_outliers(self):
        csv, truth = planted_csv()
        df, _, _ = load_numeric_csv(io.BytesIO(csv))
        result = detect_anomalies(df, Params(contamination=0.05, n_estimators=100))
        flagged = np.array(result["labels"], bool)
        self.assertGreaterEqual((flagged & truth).sum() / truth.sum(), 0.9)

    def test_contamination_controls_flagged_share(self):
        csv, _ = planted_csv()
        df, _, _ = load_numeric_csv(io.BytesIO(csv))
        for c in (0.02, 0.10, 0.20):
            result = detect_anomalies(df, Params(contamination=c))
            self.assertAlmostEqual(result["n_anomalies"] / result["n_rows"], c, delta=0.01)

    def test_scores_are_higher_for_flagged_rows(self):
        csv, _ = planted_csv()
        df, _, _ = load_numeric_csv(io.BytesIO(csv))
        result = detect_anomalies(df, Params())
        scores, labels = np.array(result["scores"]), np.array(result["labels"], bool)
        self.assertGreater(scores[labels].min(), result["threshold"] - 1e-3)
        self.assertGreater(scores[labels].mean(), scores[~labels].mean())

    def test_results_are_reproducible(self):
        csv, _ = planted_csv()
        df, _, _ = load_numeric_csv(io.BytesIO(csv))
        first, second = detect_anomalies(df, Params()), detect_anomalies(df, Params())
        self.assertEqual(first["scores"], second["scores"])

    def test_drops_missing_rows_and_ignores_text_columns(self):
        csv = b"id,x,y\nA,1,2\nB,,3\n" + b"\n".join(b"C,%d,%d" % (i, i * 2) for i in range(20))
        df, ignored, dropped = load_numeric_csv(io.BytesIO(csv))
        self.assertEqual((ignored, dropped), (["id"], 1))
        self.assertEqual(len(df), 21)

    def test_rejects_bad_input(self):
        with self.assertRaises(DetectionError):
            load_numeric_csv(io.BytesIO(b"x\n1\n2\n3\n"))               # one numeric column
        with self.assertRaises(DetectionError):
            load_numeric_csv(io.BytesIO(b"x,y\n1,2\n3,4\n"))            # too few rows
        with self.assertRaises(DetectionError):
            Params(contamination=0.9)
        with self.assertRaises(DetectionError):
            Params(n_estimators=5)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()
        self.csv, self.truth = planted_csv()

    def post(self, **fields):
        data = {"file": (io.BytesIO(self.csv), "data.csv"), **fields}
        return self.client.post("/api/detect", data=data, content_type="multipart/form-data")

    def test_index_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/healthz").get_json(), {"status": "ok"})

    def test_every_example_loads_and_is_analysable(self):
        examples = self.client.get("/api/examples").get_json()
        self.assertEqual(len(examples), 3)
        for example in examples:
            csv = self.client.get(f"/api/examples/{example['id']}")
            self.assertEqual(csv.status_code, 200)
            content = csv.data
            csv.close()
            resp = self.client.post("/api/detect", content_type="multipart/form-data",
                                    data={"file": (io.BytesIO(content), "x.csv")})
            self.assertEqual(resp.status_code, 200, example["id"])
            self.assertGreater(resp.get_json()["n_anomalies"], 0)

    def test_unknown_example_is_404(self):
        self.assertEqual(self.client.get("/api/examples/nope").status_code, 404)

    def test_detect_returns_scores_and_labels(self):
        body = self.post(contamination="0.05", n_estimators="50").get_json()
        self.assertEqual(len(body["scores"]), body["n_rows"])
        self.assertEqual(len(body["labels"]), body["n_rows"])
        self.assertEqual(body["columns"], ["a", "b"])
        self.assertEqual(sum(body["labels"]), body["n_anomalies"])

    def test_feature_subset(self):
        body = self.post(features="a").get_json()
        self.assertEqual(body["features"], ["a"])

    def test_validation_errors_are_400_json(self):
        for fields in ({"contamination": "0.9"}, {"n_estimators": "9999"},
                       {"contamination": "abc"}, {"features": "missing_col"}):
            resp = self.post(**fields)
            self.assertEqual(resp.status_code, 400, fields)
            self.assertIn("error", resp.get_json())
        resp = self.client.post("/api/detect", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_oversized_upload_is_413(self):
        big = io.BytesIO(b"x,y\n" + b"1,2\n" * (6 * 1024 * 1024 // 4))
        resp = self.client.post("/api/detect", data={"file": (big, "big.csv")},
                                content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 413)


if __name__ == "__main__":
    unittest.main()
