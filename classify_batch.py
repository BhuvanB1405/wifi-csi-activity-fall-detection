"""
Batch classifier - classifies a chunk of CSI data you've captured and saved.

HOW TO USE (same proven workflow as your data collection):
1. Run:  idf.py -p COM6 monitor
2. Perform an activity (walk / fall / stand still) for ~15-20 seconds
3. Press Ctrl+] to stop
4. Select the visible CSI_DATA text, copy it
5. Paste into a new file, e.g.:  notepad live_chunk.txt
6. Save as UTF-8 (File > Save As > Encoding: UTF-8)
7. Run:  python classify_batch.py live_chunk.txt

This prints the predicted activity for every CSI reading in that chunk,
plus an overall majority verdict - giving you a "near real-time" result
within seconds, using the exact copy-paste method that's worked all day.
"""

import sys
import re
import numpy as np
import joblib

MODEL_FILE = "trained_model.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"


def extract_features(csi_array):
    arr = np.array(csi_array, dtype=float)
    stats = [
        np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
        np.median(arr), np.sum(np.abs(arr)), np.mean(np.abs(np.diff(arr)))
    ]
    return np.concatenate([stats, arr])


def reconstruct_records(raw_text):
    flat = raw_text.replace("\r", " ").replace("\n", " ")
    parts = re.split(r'(?=CSI_DATA,)', flat)
    return [p.strip() for p in parts if p.strip().startswith("CSI_DATA")]


def parse_csi_line(line):
    try:
        match = re.search(r'\[(.*?)\]', line)
        if not match:
            return None
        csi_str = match.group(1)
        return [int(x) for x in csi_str.split(",") if x.strip() != ""]
    except (ValueError, IndexError):
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_batch.py <chunk_file.txt>")
        print("Example: python classify_batch.py live_chunk.txt")
        return

    input_file = sys.argv[1]

    print("Loading trained model...")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    print(f"Classes: {list(le.classes_)}\n")

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    records = reconstruct_records(raw_text)
    print(f"Found {len(records)} CSI readings in {input_file}\n")

    if not records:
        print("No CSI_DATA found. Make sure you pasted real monitor output.")
        return

    predictions = []
    for i, record in enumerate(records):
        csi_values = parse_csi_line(record)
        if csi_values is None or len(csi_values) not in (127, 128):
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
        predictions.append(pred_label)

        marker = "  <<< FALL DETECTED" if pred_label == "falling" else ""
        print(f"[{i:3d}] {pred_label}{marker}")

    if predictions:
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        from collections import Counter
        counts = Counter(predictions)
        for label, count in counts.most_common():
            pct = 100 * count / len(predictions)
            print(f"{label:10s}: {count:3d} readings ({pct:.0f}%)")

        overall = counts.most_common(1)[0][0]
        print(f"\nOverall verdict for this session: {overall.upper()}")


if __name__ == "__main__":
    main()
