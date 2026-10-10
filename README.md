# WiFi-CSI Human Activity and Fall Detection (ESP32)

Contactless human activity and fall detection using WiFi Channel State Information (CSI) from a single ESP32, a home router, and a Random Forest classifier. No wearables and no cameras.

This project is built on Espressif's [`esp-csi`](https://github.com/espressif/esp-csi) tool (Apache 2.0 license), specifically its `csi_recv_router` example. This repository contains my modified firmware, the Python machine-learning pipeline, the dataset, the trained model, and the full report. To rebuild the firmware, copy these files into `esp-csi/examples/get-started/csi_recv_router/`.

## How it works

1. The ESP32 connects to the home WiFi router and pings it internally (`csi_recv_router`, 100 Hz setting).
2. The firmware extracts CSI from the router's replies and prints `CSI_DATA` lines over serial. Each line holds RSSI, channel, and an array of 128 values.
3. Captured lines are saved to `.txt`, parsed into labeled CSVs, combined into one dataset, and used to train SVM and Random Forest classifiers.
4. A saved model classifies new captures from the command line or through a Streamlit dashboard.

## Hardware and software

- 1x ESP32 WROOM-32 (receiver), a 2.4 GHz home WiFi router (acts as transmitter), a Windows laptop
- ESP-IDF v5.5.5
- Python 3.11 with pandas, numpy, scipy, scikit-learn, matplotlib, joblib, streamlit

## Project files

| File | Purpose |
|---|---|
| `main/app_main.c` | Firmware. Modified to disable WiFi power-save: `esp_wifi_set_ps(WIFI_PS_NONE);` |
| `CMakeLists.txt` | Build file of the example |
| `parse_csi_log.py` | Converts a raw monitor capture (`.txt`) into a labeled CSV |
| `load_dataset.py` | Combines all labeled CSVs into `combined_dataset.csv` |
| `train_classifier.py` | Extracts features, trains SVM and Random Forest, saves the best model |
| `classify_batch.py` | Loads the saved model and classifies a capture file |
| `dashboard.py` | Streamlit dashboard that wraps `classify_batch.py` |
| `csi_logger_v2.py` | Experimental serial logger for automatic capture. Not used in the final workflow because of Windows encoding issues, so captures are copied by hand |
| `realtime_infer.py` | Experimental live-inference script. Not reliable with the router-based setup, so the project works in batch mode |
| `trained_model.pkl`, `scaler.pkl`, `label_encoder.pkl` | Saved model, scaler, and label encoder |
| `combined_dataset.csv`, `csi_*.csv` | Dataset (combined, and sample per-capture files) |
| `csi_raw_*.txt`, `person1_01.txt` | Sample raw serial captures |
| `WiFi_CSI_Fall_Detection_Report.docx` | Full project report |
| `COMMANDS.txt` | Every command used, from setup to dashboard |

## Setup

1. Install ESP-IDF v5.5.5 and open the **ESP-IDF PowerShell**.
2. Set your WiFi credentials (they are stored locally in `sdkconfig`, which is not committed):
   ```
   cd esp-csi\examples\get-started\csi_recv_router
   idf.py menuconfig
   ```
   Go to `Example Connection Configuration` and set the WiFi SSID and password.
3. Build and flash (change `COM6` to your port):
   ```
   idf.py -p COM6 flash monitor
   ```
4. Install the Python packages:
   ```
   python -m pip install pandas numpy scipy scikit-learn matplotlib joblib streamlit
   ```

## Workflow

**1. Capture data.** Run the monitor:
```
idf.py -p COM6 monitor
```
Press the ESP32 EN/RST button, wait for `got ip`, perform one activity, then copy the `CSI_DATA` lines and save them as a `.txt` file with **UTF-8** encoding.

**2. Parse a capture into a labeled CSV:**
```
python parse_csi_log.py sitstand_01.txt sitting_standing
```
For many files at once (the delay avoids filename collisions, because output names use a timestamp in seconds):
```
Get-ChildItem sitstand_*.txt | ForEach-Object { python parse_csi_log.py $_.Name sitting_standing; Start-Sleep -Seconds 1 }
```

**3. Combine all CSVs:**
```
python load_dataset.py
```

**4. Train and save the best model:**
```
python train_classifier.py
```
The classes used for training are listed in `USABLE_LABELS` in `train_classifier.py`. Add any new label there, or its data is silently dropped.

**5. Classify a capture:**
```
python classify_batch.py demo_falling.txt
```

**6. Run the dashboard:**
```
python -m streamlit run dashboard.py
```
Upload a capture `.txt` to see the verdict, class breakdown, and per-reading table.

## Results

| Stage | Classes | Samples used | Random Forest | SVM |
|---|---|---|---|---|
| Initial | 3 (empty, falling, walking) | 583 | 84.25% | 82.88% |
| Expanded | 7 (adds sitting/standing, lying down, running, bending) | 805 | 74.75% | 71.29% |
| With occupancy | 9 (adds one_person, two_person) | 1078 | 70.00% | 64.44% |

Samples per class in the 9-class dataset: falling 229, empty 220, walking 187, one_person 137, two_person 136, bending 47, sitting/standing 44, lying down 41, running 37.

The accuracy drop comes from the harder multi-class problem and from the small sample counts of the newly added classes. In the 9-class model, `lying_down` is often confused with the static `one_person` and `two_person` classes.

## Known limitations

- The home router returns only about 2-3 CSI replies per connection, whatever the ping rate. Data was collected in many short rounds, resetting the ESP32 between rounds.
- The system works in batch mode. A capture is saved first and classified afterwards, not continuously.
- Capture was automated with `pyserial` and `idf.py monitor` piping, but both hit Windows encoding problems, so captures are copied by hand and saved as UTF-8.
- Data was collected in a normally occupied household, so movement elsewhere may add some noise.
- The 4 newly added activity classes have few samples (37-47 each).

## Future work

- ESP-NOW (ESP32 to ESP32) for continuous, router-free CSI and true live operation
- Separate classifiers for activity and occupancy
- More data for the weaker classes
- FFT-based features

## Credits

- [Espressif `esp-csi`](https://github.com/espressif/esp-csi) provides the CSI firmware and tooling (Apache 2.0).
- Project work and report by Bhuvan.
