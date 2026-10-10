"""
CSI Data Logger + Live Plotter
Reads live CSI data from ESP32 (csi_recv_router) over serial,
saves it to a CSV file, and shows a live-updating graph of
CSI amplitude so you can visually see it change when someone
moves in front of the ESP32.
"""

import serial
import csv
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

# ==== CONFIGURATION - EDIT THESE ====
SERIAL_PORT = "COM6"      # Change if your ESP32 port is different
BAUD_RATE = 115200        # Default baud rate for esp-csi examples
CSV_FILENAME = f"csi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
PLOT_WINDOW = 100         # Number of recent samples to show on the live graph
# =====================================

def parse_csi_line(line):
    """
    Parses a line like:
    CSI_DATA,0,e8:6e:44:59:e1:b7,-58,11,...,"[126,96,7,0,45,-19,...]"
    Returns (timestamp, rssi, csi_values_list) or None if not a valid CSI line.
    """
    if not line.startswith("CSI_DATA"):
        return None

    try:
        # Extract the bracketed CSI array using regex
        match = re.search(r'\[(.*?)\]', line)
        if not match:
            return None
        csi_str = match.group(1)
        csi_values = [int(x) for x in csi_str.split(",") if x.strip() != ""]

        # Extract RSSI (4th comma-separated field, index 3)
        fields = line.split(",")
        rssi = int(fields[3]) if len(fields) > 3 else None

        return rssi, csi_values
    except (ValueError, IndexError):
        return None


def main():
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        # No DTR/RTS manipulation - press the physical EN/RST button on the
        # ESP32 board right after this connects, for a clean, reliable boot.
    except serial.SerialException as e:
        print(f"ERROR: Could not open {SERIAL_PORT}. Details: {e}")
        print("Check that: 1) The ESP32 is plugged in, 2) COM port is correct, "
              "3) No other program (like idf.py monitor) is using the port.")
        return

    print(f"Connected. Logging data to {CSV_FILENAME}")
    print("Press Ctrl+C to stop.\n")

    # Set up CSV file
    csv_file = open(CSV_FILENAME, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "rssi", "csi_values"])

    # Set up live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    amplitude_history = deque(maxlen=PLOT_WINDOW)
    line_plot, = ax.plot([], [])
    ax.set_xlabel("Sample number (most recent)")
    ax.set_ylabel("Mean CSI amplitude (approx)")
    ax.set_title("Live CSI Signal - move near the ESP32 to see changes")

    sample_count = 0
    raw_line_count = 0
    debug_lines_shown = 0

    try:
        while True:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue

            raw_line_count += 1

            # DEBUG: show the first 60 raw lines received, regardless of parsing
            if debug_lines_shown < 60:
                print(f"RAW[{raw_line_count}]: {raw_line[:120]}")
                debug_lines_shown += 1

            result = parse_csi_line(raw_line)
            if result is None:
                continue

            rssi, csi_values = result
            timestamp = datetime.now().isoformat()

            # Save to CSV
            csv_writer.writerow([timestamp, rssi, csi_values])
            csv_file.flush()

            # Compute a simple summary metric (mean absolute amplitude)
            if csi_values:
                mean_amp = sum(abs(v) for v in csi_values) / len(csi_values)
                amplitude_history.append(mean_amp)
                sample_count += 1

                # Update live plot every few samples for performance
                if sample_count % 5 == 0:
                    line_plot.set_data(range(len(amplitude_history)), list(amplitude_history))
                    ax.set_xlim(0, PLOT_WINDOW)
                    if amplitude_history:
                        ax.set_ylim(min(amplitude_history) - 5, max(amplitude_history) + 5)
                    plt.pause(0.001)  # yields control to GUI event loop, keeps window responsive

                if sample_count % 20 == 0:
                    print(f"[{sample_count}] RSSI: {rssi} dBm | Mean amplitude: {mean_amp:.2f}")

    except KeyboardInterrupt:
        print("\nStopping... data saved to", CSV_FILENAME)
    finally:
        ser.close()
        csv_file.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()