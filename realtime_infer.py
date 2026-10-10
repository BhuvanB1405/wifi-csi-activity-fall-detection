"""
Step 3 of the ML pipeline: Real-time inference.

Reads live_feed.txt (written by 'idf.py -p COM6 monitor > live_feed.txt'
in another window) and classifies each new CSI reading as it appears.

Uses binary file reading + manual UTF-16-LE decoding to avoid the
BOM-detection crash that happens when seeking past the start of a
UTF-16 file (which is how Windows PowerShell writes redirected output).

Press Ctrl+C to stop.
"""

import time
import re
import numpy as np
import joblib

LOG_FILE = "live_feed.txt"
MODEL_FILE = "trained_model.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
POLL_INTERVAL = 0.3


def extract_features(csi_array):
    arr = np.array(csi_array, dtype=float)
    stats = [
        np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
        np.median(arr), np.sum(np.abs(arr)), np.mean(np.abs(np.diff(arr)))
    ]
    return np.concatenate([stats, arr])


def parse_csi_line(line):
    if "CSI_DATA" not in line:
        return None
    try:
        match = re.search(r'\[(.*?)\]', line)
        if not match:
            return None
        csi_str = match.group(1)
        return [int(x) for x in csi_str.split(",") if x.strip() != ""]
    except (ValueError, IndexError):
        return None


def main():
    print("Loading trained model...")
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        le = joblib.load(LABEL_ENCODER_FILE)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find {e.filename}")
        print("Run train_classifier.py first.")
        return
    print(f"Model loaded. Classes: {list(le.classes_)}")

    try:
        f = open(LOG_FILE, "rb")
    except FileNotFoundError:
        print(f"ERROR: {LOG_FILE} not found. Start the monitor first.")
        return

    f.seek(0, 2)  # binary seek to end - no BOM issue here
    print(f"Watching {LOG_FILE}... Press Ctrl+C to stop.\n")

    prediction_count = 0
    recent_predictions = []
    lines_seen = 0
    debug_shown = 0
    leftover = ""  # holds any incomplete line between reads

    try:
        while True:
            chunk = f.read()  # read all new bytes since last read
            if not chunk:
                time.sleep(POLL_INTERVAL)
                continue

            # Decode as UTF-16-LE (no BOM required for this codec name)
            text = leftover + chunk.decode("utf-16-le", errors="ignore")
            lines = text.split("\n")
            leftover = lines[-1]  # last piece may be incomplete, save for next round
            complete_lines = lines[:-1]

            for line in complete_lines:
                lines_seen += 1
                line = line.strip()

                if debug_shown < 20 and line:
                    print(f"DEBUG raw[{lines_seen}]: {line[:100]}")
                    debug_shown += 1

                csi_values = parse_csi_line(line)
                if csi_values is None:
                    continue
                if len(csi_values) not in (127, 128):
                    continue

                features = extract_features(csi_values).reshape(1, -1)
                expected_len = scaler.mean_.shape[0]
                if features.shape[1] != expected_len:
                    if features.shape[1] < expected_len:
                        features = np.pad(features, ((0, 0), (0, expected_len - features.shape[1])))
                    else:
                        features = features[:, :expected_len]

                features_scaled = scaler.transform(features)
                pred_encoded = model.predict(features_scaled)[0]
                pred_label = le.inverse_transform([pred_encoded])[0]

                prediction_count += 1
                recent_predictions.append(pred_label)
                if len(recent_predictions) > 5:
                    recent_predictions.pop(0)
                stable_label = max(set(recent_predictions), key=recent_predictions.count)

                marker = "  <<< FALL DETECTED" if stable_label == "falling" else ""
                print(f"[{prediction_count:4d}] Instant: {pred_label:10s} | Stable: {stable_label:10s}{marker}")

    except KeyboardInterrupt:
        print(f"\nStopped. Total predictions: {prediction_count}")
    finally:
        f.close()


if __name__ == "__main__":
    main()