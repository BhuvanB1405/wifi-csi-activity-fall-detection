"""
CSI Data Logger v2 - uses idf.py monitor as a subprocess
This avoids all the DTR/RTS reset timing issues we hit trying to open
the serial port directly with pyserial. idf.py monitor (built on esptool)
already knows how to reset ESP32 boards correctly and reliably.

IMPORTANT: Run this script from INSIDE your project folder:
C:\\MyProjects\\esp-csi\\examples\\get-started\\csi_recv_router

Usage:
    python csi_logger_v2.py
"""

import subprocess
import csv
import re
import sys
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

# ==== CONFIGURATION ====
COM_PORT = "COM6"
CSV_FILENAME = f"csi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
PLOT_WINDOW = 100
# ========================

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


def main():
    print(f"Launching 'idf.py -p {COM_PORT} monitor' as subprocess...")
    print("Make sure you're running this from the csi_recv_router project folder!\n")

    try:
        process = subprocess.Popen(
            ["idf.py", "-p", COM_PORT, "monitor", "--disable-auto-color"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
            shell=True
        )
    except Exception as e:
        print(f"ERROR launching idf.py monitor: {e}")
        print("Make sure you're in the ESP-IDF CMD environment (not regular CMD)")
        print("and in the csi_recv_router project folder.")
        return

    csv_file = open(CSV_FILENAME, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "rssi", "csi_values"])
    print(f"Logging CSI data to {CSV_FILENAME}")
    print("Press Ctrl+C to stop.\n")

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    amplitude_history = deque(maxlen=PLOT_WINDOW)
    line_plot, = ax.plot([], [])
    ax.set_xlabel("Sample number (most recent)")
    ax.set_ylabel("Mean CSI amplitude (approx)")
    ax.set_title("Live CSI Signal - move near the ESP32 to see changes")

    sample_count = 0

    try:
        for raw_line in process.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if "CSI_DATA" not in raw_line and sample_count == 0:
                print(raw_line[:150])

            result = parse_csi_line(raw_line)
            if result is None:
                continue

            rssi, csi_values = result
            timestamp = datetime.now().isoformat()
            csv_writer.writerow([timestamp, rssi, csi_values])
            csv_file.flush()

            if csi_values:
                mean_amp = sum(abs(v) for v in csi_values) / len(csi_values)
                amplitude_history.append(mean_amp)
                sample_count += 1

                if sample_count % 5 == 0:
                    line_plot.set_data(range(len(amplitude_history)), list(amplitude_history))
                    ax.set_xlim(0, PLOT_WINDOW)
                    if amplitude_history:
                        ax.set_ylim(min(amplitude_history) - 5, max(amplitude_history) + 5)
                    plt.pause(0.001)

                if sample_count % 20 == 0:
                    print(f"[{sample_count}] RSSI: {rssi} dBm | Mean amplitude: {mean_amp:.2f}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        process.terminate()
        csv_file.close()
        print(f"Data saved to {CSV_FILENAME}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()