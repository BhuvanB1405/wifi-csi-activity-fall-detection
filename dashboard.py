"""
WiFi-CSI Activity & Fall Detection — Streamlit Dashboard

This dashboard wraps your existing, already-working classify_batch.py
instead of reimplementing any CSI parsing or feature extraction logic.
That guarantees the dashboard always classifies exactly the same way
your proven command-line pipeline does — no risk of drift between the two.

SETUP:
1. Copy this file into the SAME folder as classify_batch.py, trained_model.pkl,
   scaler.pkl, and label_encoder.pkl:
   C:\\MyProjects\\esp-csi\\examples\\get-started\\csi_recv_router

2. Install Streamlit if you haven't already:
   pip install streamlit

3. Run it:
   streamlit run dashboard.py

   This opens a browser window with the dashboard. Leave the terminal
   running while you use it — closing the terminal closes the dashboard.

USAGE:
Upload any raw CSI capture .txt file (the same kind you've been saving
throughout this project, e.g. test_falling_live.txt) and it will run
classify_batch.py on it and display the result visually.
"""

import streamlit as st
import subprocess
import sys
import re
import os
import pandas as pd

st.set_page_config(page_title="WiFi-CSI Fall Detection", page_icon="📡", layout="centered")

st.title("📡 WiFi-CSI Activity & Fall Detection")
st.caption(
    "Upload a raw CSI capture (.txt) to classify it. This runs your existing "
    "classify_batch.py under the hood — same model, same logic, just a visual layer on top."
)

# Colors for the verdict card, grouped by rough meaning rather than one color per class
VERDICT_COLORS = {
    "falling": "#e74c3c",       # red — the one that matters most
    "empty": "#2ecc71",         # green — nothing happening
    "one_person": "#3498db",    # blue — presence detected
    "two_person": "#3498db",
}
DEFAULT_COLOR = "#5865f2"  # blue-purple for any other activity class

uploaded_file = st.file_uploader("Upload a CSI capture file (.txt)", type=["txt"])

if uploaded_file is not None:
    temp_filename = f"_dashboard_upload_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Running classify_batch.py..."):
        try:
            result = subprocess.run(
                [sys.executable, "classify_batch.py", temp_filename],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
        except FileNotFoundError:
            output = ""
            st.error(
                "Couldn't find classify_batch.py in this folder. "
                "Make sure dashboard.py is saved in the same folder as classify_batch.py."
            )

    if output:
        readings_match = re.search(r"Found (\d+) CSI readings", output)
        num_readings = int(readings_match.group(1)) if readings_match else None

        if num_readings is None:
            st.error("No CSI_DATA found in this file. Make sure you pasted real monitor output.")
            with st.expander("Raw output"):
                st.code(output)
        else:
            reading_lines = re.findall(r"\[\s*(\d+)\]\s+(\S+)", output)
            summary_lines = re.findall(r"(\w+)\s*:\s*(\d+) readings \((\d+)%\)", output)
            verdict_match = re.search(r"Overall verdict for this session:\s*(\S+)", output)
            overall_verdict = verdict_match.group(1) if verdict_match else None

            st.subheader(f"Found {num_readings} CSI reading(s)")

            if overall_verdict:
                color = VERDICT_COLORS.get(overall_verdict.lower(), DEFAULT_COLOR)
                st.markdown(
                    f"""
                    <div style="background-color:{color}; padding: 28px; border-radius: 14px;
                                text-align:center; margin: 12px 0 24px;">
                        <div style="color:white; font-size:15px; opacity:0.85;">OVERALL VERDICT</div>
                        <div style="color:white; font-size:32px; font-weight:700; margin-top:4px;">
                            {overall_verdict.upper()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if summary_lines:
                df_summary = pd.DataFrame(summary_lines, columns=["Class", "Count", "Percent"])
                df_summary["Percent"] = df_summary["Percent"].astype(int)
                st.markdown("**Class breakdown**")
                st.bar_chart(df_summary.set_index("Class")["Percent"])

            if reading_lines:
                st.markdown("**Per-reading detail**")
                df_readings = pd.DataFrame(reading_lines, columns=["Reading #", "Predicted class"])
                st.dataframe(df_readings, use_container_width=True, hide_index=True)

            with st.expander("Raw classify_batch.py output"):
                st.code(output)

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

else:
    st.info("Upload a .txt CSI capture file to get started — e.g. one of your saved test_*.txt files.")

st.divider()
st.caption("WiFi-CSI Human Activity & Fall Detection · Random Forest classifier")
