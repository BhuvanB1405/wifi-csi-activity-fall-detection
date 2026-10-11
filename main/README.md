   # Firmware (main)

   `app_main.c` is the modified `csi_recv_router` firmware from Espressif's esp-csi.

   Change made: WiFi power-save is disabled with `esp_wifi_set_ps(WIFI_PS_NONE);` so the ESP32 receives CSI packets reliably.

   Copy this folder into `esp-csi/examples/get-started/csi_recv_router/` and build with ESP-IDF v5.5.5.
