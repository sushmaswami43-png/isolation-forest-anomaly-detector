"""Flask service: serves the UI and a JSON API around the Isolation Forest."""
import os

from flask import Flask, jsonify, render_template, request, send_from_directory

from detector import DetectionError, Params, detect_anomalies, load_numeric_csv

EXAMPLES = {
    "transactions": ("transactions.csv",
                     "Card transactions: amount vs. seconds since previous purchase (fraud-style outliers)"),
    "network": ("network_traffic.csv",
                "Network flows: bytes, packets/s and duration (traffic-spike / scan outliers)"),
    "sensors": ("sensor_quality.csv",
                "Factory sensors: temperature, vibration and pressure (faulty-part outliers)"),
}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    @app.get("/api/examples")
    def list_examples():
        return jsonify([{"id": key, "description": desc} for key, (_, desc) in EXAMPLES.items()])

    @app.get("/api/examples/<name>")
    def get_example(name):
        if name not in EXAMPLES:
            return jsonify(error="Unknown example dataset."), 404
        return send_from_directory("examples", EXAMPLES[name][0], mimetype="text/csv")

    @app.post("/api/detect")
    def detect():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise DetectionError("Please upload a CSV file.")
        try:
            contamination = float(request.form.get("contamination", 0.05))
            n_estimators = int(request.form.get("n_estimators", 100))
        except ValueError:
            raise DetectionError("contamination and n_estimators must be numbers.") from None
        params = Params(contamination=contamination, n_estimators=n_estimators)

        features = [f for f in request.form.get("features", "").split(",") if f]
        df, ignored, dropped = load_numeric_csv(upload.stream)
        result = detect_anomalies(df, params, features or None)
        result.update(ignored_columns=ignored, dropped_rows=dropped)
        return jsonify(result)

    @app.errorhandler(DetectionError)
    def handle_detection_error(err):
        return jsonify(error=str(err)), 400

    @app.errorhandler(413)
    def handle_too_large(_):
        return jsonify(error="File is too large (limit 5 MB)."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
