"""
CSI Log Parser - reads a saved idf.py monitor output file and extracts
CSI data into a clean CSV, ready for analysis/ML.

HOW TO USE:
1. Record data first:
   idf.py -p COM6 monitor > csi_raw_walking.txt
   (let it run while you perform an activity, then Ctrl+C to stop)

2. Parse it:
   python parse_csi_log.py csi_raw_walking.txt walking

   This creates: csi_walking_<timestamp>.csv
"""

import csv
import re
import sys
from datetime import datetime


def parse_csi_line(line):
    if "CSI_DATA" not in line:
        return None
    try:
        match = re.search(r'\[(.*?)\]', line)
        if not match:
            return None
        csi_str = match.group(1)
        csi_values = [int(x) for x in csi_str.split(",") if x.strip() != ""]

        pre_bracket = line.split('"')[0] if '"' in line else line.split('[')[0]
        fields = pre_bracket.strip(",").split(",")
        rssi = None
        for f in fields:
            f = f.strip()
            if f.lstrip("-").isdigit() and f.startswith("-"):
                rssi = int(f)
                break
        return rssi, csi_values
    except (ValueError, IndexError):
        return None


def reconstruct_records(raw_text):
    """
    Terminal copy-paste often breaks single CSI_DATA lines across multiple
    physical lines (due to word-wrap). This stitches everything back into
    one continuous stream, then splits on 'CSI_DATA' markers to recover
    complete records regardless of where line breaks fell.
    """
    # Remove all newlines/carriage returns to undo any wrapping
    flat = raw_text.replace("\r", " ").replace("\n", " ")
    # Split right before each "CSI_DATA" occurrence
    parts = re.split(r'(?=CSI_DATA,)', flat)
    records = [p.strip() for p in parts if p.strip().startswith("CSI_DATA")]
    return records


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_csi_log.py <input_log_file.txt> [activity_label]")
        print("Example: python parse_csi_log.py csi_raw_walking.txt walking")
        return

    input_file = sys.argv[1]
    activity_label = sys.argv[2] if len(sys.argv) > 2 else "unlabeled"
    output_file = f"csi_{activity_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Reading {input_file}...")

    try:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"ERROR: Could not find file '{input_file}'")
        print("Make sure you're running this from the same folder as the log file,")
        print("or provide the full path.")
        return

    records = reconstruct_records(raw_text)
    print(f"Reconstructed {len(records)} CSI_DATA records. Parsing...")

    # DEBUG: show exactly what's in the file if nothing was found
    if len(records) == 0:
        print("\n--- DEBUG: First 500 characters of raw file content ---")
        print(repr(raw_text[:500]))
        print("--- END DEBUG ---\n")
        print(f"Does 'CSI_DATA' appear anywhere in file? {'CSI_DATA' in raw_text}")
        print(f"File length: {len(raw_text)} characters")

    rows_written = 0
    with open(output_file, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["index", "activity_label", "rssi", "csi_values"])

        for record in records:
            result = parse_csi_line(record)
            if result is None:
                continue
            rssi, csi_values = result
            writer.writerow([rows_written, activity_label, rssi, csi_values])
            rows_written += 1

    print(f"\nDone! Extracted {rows_written} CSI samples.")
    print(f"Saved to: {output_file}")

    if rows_written == 0:
        print("\nWARNING: No CSI data found in the log file.")
        print("Check that the log file actually contains lines starting with 'CSI_DATA'.")


if __name__ == "__main__":
    main()