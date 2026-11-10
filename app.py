"""
app.py - Industrial Quality Inspection Dashboard
Label Alignment & OCR Expiration Date Validator (Streamlit PoC)
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
import time
import os
from PIL import Image
from io import BytesIO

# Import custom inspection and database modules
import importlib
import inspector as inspector_module
importlib.reload(inspector_module)
from inspector import IndustrialInspector
import database as db
importlib.reload(db)
from video_stream import FreshFrameStreamer
from hardware_io import IndustrialActuatorController

# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Industrial QA - Label & OCR Validator",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial CSS Theme
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #303e4d;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #90a4ae;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        margin: 5px 0;
        color: #eceff1;
    }
    .badge-pass {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: #ffffff;
        font-size: 32px;
        font-weight: 900;
        padding: 12px;
        border-radius: 8px;
        letter-spacing: 2px;
        text-align: center;
        border: 2px solid #4caf50;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.4);
    }
    .badge-reject {
        background: linear-gradient(135deg, #c62828 0%, #b71c1c 100%);
        color: #ffffff;
        font-size: 32px;
        font-weight: 900;
        padding: 12px;
        border-radius: 8px;
        letter-spacing: 2px;
        text-align: center;
        border: 2px solid #ef5350;
        box-shadow: 0 0 15px rgba(239, 83, 80, 0.4);
    }
    .badge-standby {
        background: #263238;
        color: #90a4ae;
        font-size: 28px;
        font-weight: 700;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #455a64;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("🏭 QA Station Controls")

# Input Mode Selector
input_mode = st.sidebar.selectbox(
    "Select Video / Image Input",
    [
        "IP Camera / Smartphone Stream",
        "USB / Built-in Webcam (Index 0)",
        "Offline File Upload",
        "Demo Samples (Pass / Fail PoC)"
    ]
)

# Stream Settings
stream_url = ""
if input_mode == "IP Camera / Smartphone Stream":
    stream_url = st.sidebar.text_input(
        "Smartphone Camera URL",
        value="http://192.168.1.100:8080/video",
        help="รองรับ Local Wi-Fi (http://192.168.x.x:8080/video) หรือ Tailscale VPN IP (http://100.x.y.z:8080/video)"
    )
    with st.sidebar.expander("💡 แนะนำ: เชื่อมต่อผ่าน Tailscale (4G/5G)"):
        st.caption(
            "หาก Wi-Fi โรงงานบล็อก (AP Isolation) หรือมือถือใช้เน็ต 4G/5G ให้ติดตั้ง **Tailscale** "
            "ทั้งบนมือถือและคอมพิวเตอร์ จะได้ IP วง 100.x.y.z จากนั้นใส่ URL เช่น:\n"
            "`http://100.x.y.z:8080/video`"
        )
elif input_mode == "USB / Built-in Webcam (Index 0)":
    stream_url = "0"

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Inspection Thresholds")

# Skew Angle Threshold
skew_limit = st.sidebar.slider(
    "Max Allowed Skew (± Degrees)",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.5,
    help="Labels tilted beyond this angle are marked as REJECT"
)

# Factory Automation Options
st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Factory Automation & Alarms")
sound_alarm = st.sidebar.checkbox("🔊 Factory Audio Buzzer on Reject", value=True)
reject_pulse_ms = st.sidebar.slider("Pneumatic Ejector Pulse (ms)", 50, 500, 150, 25)

# ROI Configuration
with st.sidebar.expander("🎯 OCR Region of Interest (ROI) Window"):
    roi_mode = st.radio(
        "Target ROI Zone",
        ["Full Frame", "Bottom 40% (Standard Date Zone)", "Custom Bounding Window"],
        index=0
    )
    custom_y_range = (60, 100)
    custom_x_range = (0, 100)
    if roi_mode == "Custom Bounding Window":
        custom_y_range = st.slider("Vertical Window (Y %)", 0, 100, (50, 100))
        custom_x_range = st.slider("Horizontal Window (X %)", 0, 100, (0, 100))

# Edge Detection & Line Tuning
with st.sidebar.expander("🛠️ Advanced OpenCV Filter Settings"):
    canny1 = st.slider("Canny Min Threshold", 10, 200, 50, 10)
    canny2 = st.slider("Canny Max Threshold", 50, 300, 150, 10)
    hough_thresh = st.slider("Hough Accumulator Thresh", 20, 150, 50, 5)
    min_line_len = st.slider("Min Line Length (px)", 20, 200, 60, 10)
    dot_matrix = st.checkbox("Dot-Matrix Inkjet Dilation Filter", value=True)
    dilation_k = st.slider("Dilation Kernel Size", 1, 5, 2)

# Instantiate Inspector Engine
inspector = IndustrialInspector(
    skew_threshold=skew_limit,
    canny_thresh1=canny1,
    canny_thresh2=canny2,
    hough_threshold=hough_thresh,
    min_line_length=min_line_len,
    enable_dot_matrix_filter=dot_matrix,
    dilation_kernel_size=dilation_k
)

def get_active_roi(img_shape):
    """Calculates ROI pixel coordinates from sidebar configuration."""
    h, w = img_shape[:2]
    if roi_mode == "Bottom 40% (Standard Date Zone)":
        return (int(h * 0.60), h, 0, w)
    elif roi_mode == "Custom Bounding Window":
        ymin = int(h * custom_y_range[0] / 100.0)
        ymax = int(h * custom_y_range[1] / 100.0)
        xmin = int(w * custom_x_range[0] / 100.0)
        xmax = int(w * custom_x_range[1] / 100.0)
        return (ymin, ymax, xmin, xmax)
    return None

# Session State Initializations
if "streamer" not in st.session_state:
    st.session_state.streamer = None
if "last_inspection" not in st.session_state:
    st.session_state.last_inspection = None
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False
if "last_auto_trigger" not in st.session_state:
    st.session_state.last_auto_trigger = 0.0
if "actuator" not in st.session_state:
    st.session_state.actuator = IndustrialActuatorController(ejector_pulse_ms=reject_pulse_ms)
else:
    st.session_state.actuator.ejector_pulse_ms = reject_pulse_ms

# ---------------------------------------------------------
# Main Page Header & KPI Cards
# ---------------------------------------------------------
st.title("Industrial Visual Quality Inspection")
st.caption("PoC System: Packaging Label Alignment & Expiration Date (OCR) Automated Verification")

# Top KPI Metric Banners
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns([1.2, 1.2, 1.6, 1.0])

last_res = st.session_state.last_inspection

with kpi_col1:
    if last_res:
        skew_deg = last_res.get("skew_angle", 0.0)
        st_ang = last_res.get("angle_status", "N/A")
        delta_color = "normal" if st_ang == "PASS" else "inverse"
        st.metric(
            label="Label Skew Angle",
            value=f"{skew_deg:+.2f}°",
            delta=f"{st_ang} (Limit: ±{skew_limit}°)",
            delta_color=delta_color
        )
    else:
        st.metric(label="Label Skew Angle", value="-- °", delta="STANDBY")

with kpi_col2:
    if last_res:
        dt_str = last_res.get("date_found") or "NOT DETECTED"
        st_dt = last_res.get("ocr_status", "N/A")
        delta_color = "normal" if st_dt == "PASS" else "inverse"
        st.metric(
            label="EXP / MFG Date",
            value=dt_str,
            delta=f"FORMAT {st_dt}",
            delta_color=delta_color
        )
    else:
        st.metric(label="EXP / MFG Date", value="--", delta="STANDBY")

with kpi_col3:
    if last_res:
        verdict = last_res.get("verdict", "STANDBY")
        if verdict == "PASS":
            st.markdown('<div class="badge-pass">✅ PASS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-reject">❌ REJECT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-standby">READY</div>', unsafe_allow_html=True)

with kpi_col4:
    if last_res:
        c_time = last_res.get("cycle_time_ms", 0.0)
        st.metric(label="Cycle Time", value=f"{c_time:.0f} ms")
    else:
        st.metric(label="Cycle Time", value="-- ms")

# Medicine Sachet (ซองยา) Detailed Metadata Card
if last_res and "sachet_info" in last_res:
    s_info = last_res["sachet_info"]
    st.markdown("""
    <div style="background-color: #1a222d; border: 1px solid #2d3b4d; border-radius: 8px; padding: 12px 18px; margin-top: 10px;">
        <span style="font-weight: 700; color: #4fc3f7; font-size: 15px;">💊 ข้อมูลจำเพาะและรายละเอียดบนซองยา (Medicine Sachet Specification):</span>
    </div>
    """, unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("ชื่อตัวยา (Drug)", s_info.get("drug_name", "-"))
    m_col2.metric("ขนาดความแรง (Dosage)", s_info.get("dosage", "-"))
    m_col3.metric("รุ่นการผลิต (LOT)", s_info.get("lot_no", "-"))
    m_col4.metric("วันผลิต (MFG)", s_info.get("mfg_date", "-"))
    m_col5.metric("วันหมดอายุ (EXP)", s_info.get("exp_date", "-"))
    m_col6.metric("ทะเบียนยา (Reg. No.)", s_info.get("reg_no", "-"))

    # Pharmaceutical Sanity & Master Data Comparison Banner
    sanity_st = last_res.get("sanity_status", "PASS")
    defect_tp = last_res.get("defect_type", "NONE")
    calc_sl = last_res.get("calculated_shelf_life")
    exp_sl = last_res.get("expected_shelf_life")
    reasons = last_res.get("sanity_reasons", [])

    if sanity_st == "PASS":
        sl_text = f" | Shelf-Life: **{calc_sl} เดือน** (ตรงกับ Master Data: {exp_sl} เดือน)" if calc_sl else ""
        st.success(f"🛡️ **GMP Data Sanity: PASS**{sl_text} | ไม่พบปัญหาฉลากสลับรุ่น (No Label Mix-up)")
    else:
        reason_str = f" : {' • '.join(reasons)}" if reasons else ""
        st.error(f"🚨 **GMP QUALITY DEFECT: {defect_tp}**{reason_str} -> ระบบตัดสิน REJECT อัตโนมัติ!")

    with st.expander("🔍 ดูข้อความดิบทั้งหมดที่อ่านได้จากภาพสด (Raw Scanned OCR Text)"):
        st.code(last_res.get("raw_text", "No text detected"), language="text")

# Factory Andon Light & Pneumatic Solenoid Ejector Actuation Bar
if last_res:
    verdict = last_res.get("verdict", "STANDBY")
    act_res = st.session_state.actuator.trigger_verdict_actuation(verdict, last_res)

    if verdict == "REJECT":
        andon_html = f"""
        <div style="background: #2a1215; border: 1px solid #c62828; border-radius: 6px; padding: 10px 14px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="color: #ef5350; font-weight: 800; font-size: 14px;">🔴 ANDON TOWER: RED REJECT ALARM</span>
                <span style="color: #ffcdd2; margin-left: 12px; font-size: 13px;">{act_res['action_desc']}</span>
            </div>
            <div style="background: #b71c1c; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px;">
                TOTAL EJECTED: {act_res['total_ejected']}
            </div>
        </div>
        """
        st.markdown(andon_html, unsafe_allow_html=True)
        if sound_alarm:
            components.html("""
            <script>
            try {
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                if (AudioCtx) {
                    const ctx = new AudioCtx();
                    if (ctx.state === 'suspended') { ctx.resume(); }
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(650, ctx.currentTime);
                    gain.gain.setValueAtTime(0.18, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start();
                    osc.stop(ctx.currentTime + 0.35);
                }
            } catch(e) {}
            </script>
            """, height=0, width=0)
    else:
        andon_html = f"""
        <div style="background: #102416; border: 1px solid #2e7d32; border-radius: 6px; padding: 10px 14px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="color: #81c784; font-weight: 800; font-size: 14px;">🟢 ANDON TOWER: GREEN NORMAL</span>
                <span style="color: #c8e6c9; margin-left: 12px; font-size: 13px;">{act_res['action_desc']}</span>
            </div>
            <div style="background: #1b5e20; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px;">
                TOTAL EJECTED: {act_res['total_ejected']}
            </div>
        </div>
        """
        st.markdown(andon_html, unsafe_allow_html=True)

st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Two-Column Visual Display
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📹 Station Camera Feed / Inspection Target")
    feed_placeholder = st.empty()

with col_right:
    st.subheader("🔬 Vision Analysis (Edges & OCR Overlay)")
    analysis_placeholder = st.empty()

# ---------------------------------------------------------
# Handling Execution Modes
# ---------------------------------------------------------

# MODE 1 & 2: LIVE STREAMING (IP Camera or Webcam)
if input_mode in ["IP Camera / Smartphone Stream", "USB / Built-in Webcam (Index 0)"]:
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 2])
    with c_btn1:
        start_btn = st.button("▶️ Start Live Stream", use_container_width=True)
    with c_btn2:
        stop_btn = st.button("⏹️ Stop Stream", use_container_width=True)
    with c_btn3:
        inspect_trigger = st.button("📸 Inspect Current Frame (Run Full QA)", type="primary", use_container_width=True)

    # Conveyor Automation Mode (Auto-Trigger)
    c_auto1, c_auto2 = st.columns([1, 2])
    with c_auto1:
        auto_inspect = st.checkbox("🤖 Auto-Inspect (Conveyor Trigger)", value=False, help="จำลอง Optical Sensor ตรวจจับและบันทึกผลอัตโนมัติตามช่วงเวลา")
    with c_auto2:
        auto_interval = st.slider("Trigger Interval (Seconds)", 1.0, 10.0, 3.0, 0.5, disabled=not auto_inspect)

    if start_btn:
        if st.session_state.streamer:
            st.session_state.streamer.stop()
        st.session_state.streamer = FreshFrameStreamer(stream_url)
        if st.session_state.streamer.start():
            st.session_state.is_streaming = True
            st.toast("Connected to camera stream successfully!", icon="✅")
        else:
            st.error(f"Could not connect to stream at: {stream_url}. Please check IP/port and network.")
            st.session_state.is_streaming = False

    if stop_btn:
        if st.session_state.streamer:
            st.session_state.streamer.stop()
        st.session_state.streamer = None
        st.session_state.is_streaming = False
        st.info("Stream stopped.")

    # Live Processing Loop
    if st.session_state.is_streaming and st.session_state.streamer:
        ret, frame = st.session_state.streamer.read()
        if ret and frame is not None:
            # High-speed Skew angle calculation (Runs in ~3ms per frame)
            skew_angle, angle_st, skew_overlay, edges, l_cnt = inspector.measure_skew_angle(frame)

            # Display Real-time live frame
            feed_placeholder.image(cv2.cvtColor(skew_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            analysis_placeholder.image(edges, caption="Live Canny Edge Detection", use_container_width=True)

            # Trigger Condition: Manual click or Automated conveyor trigger
            now = time.time()
            trigger_condition = inspect_trigger or (auto_inspect and (now - st.session_state.last_auto_trigger >= auto_interval))

            if trigger_condition:
                st.session_state.last_auto_trigger = now
                with st.spinner("Running high-precision OCR & Full Quality Validation..."):
                    active_roi = get_active_roi(frame.shape)
                    result = inspector.inspect_frame(frame, roi_bbox=active_roi)
                    s_inf = result.get("sachet_info", {})
                    # Log into SQLite DB
                    db.log_inspection(
                        skew_angle=result["skew_angle"],
                        angle_status=result["angle_status"],
                        date_found=result["date_found"],
                        ocr_status=result["ocr_status"],
                        verdict=result["verdict"],
                        raw_text=result["raw_text"],
                        cycle_time_ms=result["cycle_time_ms"],
                        drug_name=s_inf.get("drug_name", "-"),
                        dosage=s_inf.get("dosage", "-"),
                        lot_no=s_inf.get("lot_no", "-"),
                        mfg_date=s_inf.get("mfg_date", "-"),
                        reg_no=s_inf.get("reg_no", "-"),
                        sanity_status=result.get("sanity_status", "PASS"),
                        defect_type=result.get("defect_type", "NONE"),
                        master_match=result.get("master_match", "MATCHED")
                    )
                    st.session_state.last_inspection = result
                    st.rerun()
        else:
            feed_placeholder.warning("Waiting for next video frame...")


# MODE 3: OFFLINE FILE UPLOAD
elif input_mode == "Offline File Upload":
    uploaded_file = st.file_uploader("Upload Packaging Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is not None:
            feed_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Source Image", use_container_width=True)

            if st.button("🚀 Run Full Inspection", type="primary"):
                with st.spinner("Processing Skew Angle & Date OCR..."):
                    active_roi = get_active_roi(img.shape)
                    result = inspector.inspect_frame(img, roi_bbox=active_roi)
                    s_inf = result.get("sachet_info", {})
                    db.log_inspection(
                        skew_angle=result["skew_angle"],
                        angle_status=result["angle_status"],
                        date_found=result["date_found"],
                        ocr_status=result["ocr_status"],
                        verdict=result["verdict"],
                        raw_text=result["raw_text"],
                        cycle_time_ms=result["cycle_time_ms"],
                        drug_name=s_inf.get("drug_name", "-"),
                        dosage=s_inf.get("dosage", "-"),
                        lot_no=s_inf.get("lot_no", "-"),
                        mfg_date=s_inf.get("mfg_date", "-"),
                        reg_no=s_inf.get("reg_no", "-"),
                        sanity_status=result.get("sanity_status", "PASS"),
                        defect_type=result.get("defect_type", "NONE"),
                        master_match=result.get("master_match", "MATCHED")
                    )
                    st.session_state.last_inspection = result
                    st.rerun()

    if last_res:
        analysis_placeholder.image(
            cv2.cvtColor(last_res["annotated_image"], cv2.COLOR_BGR2RGB),
            caption=f"Inspection Result: {last_res['verdict']}",
            use_container_width=True
        )


# MODE 4: DEMO SAMPLES (Built-in Synthetic Tests)
elif input_mode == "Demo Samples (Pass / Fail PoC)":
    # Ensure sample generator exists
    import generate_sample
    generate_sample.generate_all()

    sample_choice = st.radio(
        "Choose Pre-configured Industrial Test Case (Packaging Boxes & Medicine Sachets):",
        [
            "💊 1. Sachet PASS: Paracetamol 500mg, Straight Seal (+0.4°), EXP (11/03/2029)",
            "💊 2. Sachet FAIL Skew: Paracetamol 500mg, Tilted Seal (+5.8° > 3.0°)",
            "💊 3. Sachet FAIL Date: Paracetamol 500mg, Corrupt / Unreadable EXP Date",
            "📦 4. Box Label PASS: Straight (+0.5°), EXP (24/11/2026)",
            "📦 5. Box Label FAIL Skew: Tilted (+6.5° > 3.0°)",
            "📦 6. Box Label FAIL OCR: Corrupt Date Stamp"
        ],
        horizontal=False
    )

    sample_map = {
        "💊 1. Sachet PASS: Paracetamol 500mg, Straight Seal (+0.4°), EXP (11/03/2029)": "sachet_pass.jpg",
        "💊 2. Sachet FAIL Skew: Paracetamol 500mg, Tilted Seal (+5.8° > 3.0°)": "sachet_fail_skew.jpg",
        "💊 3. Sachet FAIL Date: Paracetamol 500mg, Corrupt / Unreadable EXP Date": "sachet_fail_exp.jpg",
        "📦 4. Box Label PASS: Straight (+0.5°), EXP (24/11/2026)": "sample_pass.jpg",
        "📦 5. Box Label FAIL Skew: Tilted (+6.5° > 3.0°)": "sample_fail_skew.jpg",
        "📦 6. Box Label FAIL OCR: Corrupt Date Stamp": "sample_fail_ocr.jpg"
    }

    sample_path = os.path.join(os.path.dirname(__file__), "samples", sample_map[sample_choice])
    sample_img = cv2.imread(sample_path)

    feed_placeholder.image(
        cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB),
        caption=f"Selected PoC Sample: {sample_map[sample_choice]}",
        use_container_width=True
    )

    if st.button("⚡ Inspect Demo Sample", type="primary"):
        with st.spinner("Inspecting Sample Image..."):
            active_roi = get_active_roi(sample_img.shape)
            result = inspector.inspect_frame(sample_img, roi_bbox=active_roi)
            s_inf = result.get("sachet_info", {})
            db.log_inspection(
                skew_angle=result["skew_angle"],
                angle_status=result["angle_status"],
                date_found=result["date_found"],
                ocr_status=result["ocr_status"],
                verdict=result["verdict"],
                raw_text=result["raw_text"],
                cycle_time_ms=result["cycle_time_ms"],
                drug_name=s_inf.get("drug_name", "-"),
                dosage=s_inf.get("dosage", "-"),
                lot_no=s_inf.get("lot_no", "-"),
                mfg_date=s_inf.get("mfg_date", "-"),
                reg_no=s_inf.get("reg_no", "-"),
                sanity_status=result.get("sanity_status", "PASS"),
                defect_type=result.get("defect_type", "NONE"),
                master_match=result.get("master_match", "MATCHED")
            )
            st.session_state.last_inspection = result
            st.rerun()

    if last_res:
        analysis_placeholder.image(
            cv2.cvtColor(last_res["annotated_image"], cv2.COLOR_BGR2RGB),
            caption=f"Overlay Result: {last_res['verdict']}",
            use_container_width=True
        )


# ---------------------------------------------------------
# Database & Historical Log Table
# ---------------------------------------------------------
st.markdown("<hr style='margin: 25px 0;'>", unsafe_allow_html=True)
st.subheader("📋 Production Quality History & Log (SQLite)")

stats = db.get_statistics()
s_col1, s_col2, s_col3, s_col4 = st.columns(4)
s_col1.metric("Total Inspected", stats["total"])
s_col2.metric("Passed Items", stats["passed"])
s_col3.metric("Rejected Items", stats["rejected"])
s_col4.metric("First-Pass Yield", f"{stats['yield_rate']}%")

# Recent inspections
recent_df = db.get_all_logs_df()

if not recent_df.empty:
    # Color-code dataframe styling
    def highlight_verdict(val):
        color = '#2e7d32' if val == 'PASS' else ('#c62828' if val == 'REJECT' else '')
        return f'background-color: {color}; color: white; font-weight: bold;'

    try:
        styled_df = recent_df.head(10).style.map(highlight_verdict, subset=['verdict'])
    except AttributeError:
        styled_df = recent_df.head(10).style.applymap(highlight_verdict, subset=['verdict'])
    st.dataframe(styled_df, use_container_width=True)

    # CSV Export Button
    csv_bytes = recent_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Quality Report (CSV)",
        data=csv_bytes,
        file_name="industrial_qa_inspection_report.csv",
        mime="text/csv"
    )

    # ---------------------------------------------------------
    # SPC (Statistical Process Control) & Pareto Analytics
    # ---------------------------------------------------------
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
    st.subheader("📈 Statistical Process Control (SPC) & Defect Analytics")

    c_spc1, c_spc2 = st.columns([1.5, 1])

    with c_spc1:
        st.markdown("**Label Skew Angle Run Chart (vs Tolerance Limits)**")
        # Prepare data for SPC run chart
        spc_df = recent_df.copy().sort_values(by="id")
        spc_df["Upper Limit (+USL)"] = skew_limit
        spc_df["Lower Limit (-LSL)"] = -skew_limit
        spc_df["Target (0°)"] = 0.0

        chart_data = spc_df.set_index("id")[["skew_angle", "Upper Limit (+USL)", "Lower Limit (-LSL)", "Target (0°)"]]
        st.line_chart(chart_data, color=["#29b6f6", "#ef5350", "#ef5350", "#78909c"])

    with c_spc2:
        st.markdown("**Defect Mode Breakdown (Pareto)**")
        skew_fails = (recent_df["angle_status"] == "FAIL").sum()
        ocr_fails = (recent_df["ocr_status"] == "FAIL").sum()
        total_defects = skew_fails + ocr_fails

        if total_defects > 0:
            defect_df = pd.DataFrame({
                "Defect Category": ["Skew Alignment (> ±Limit)", "Date OCR / Corrupt Stamp"],
                "Defect Count": [int(skew_fails), int(ocr_fails)]
            })
            st.bar_chart(defect_df.set_index("Defect Category"), color="#ff7043")
        else:
            st.success("🎉 Zero defects recorded in current production batch!")

    # ---------------------------------------------------------
    # Master Data Catalog & Industrial Architecture Guide
    # ---------------------------------------------------------
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
    c_inf1, c_inf2 = st.columns(2)

    with c_inf1:
        with st.expander("📋 ตารางข้อมูลยามาตรฐาน (Master Product Catalog - product_masters)", expanded=False):
            masters_data = db.get_product_masters()
            if masters_data:
                m_df = pd.DataFrame(masters_data)[["reg_no", "drug_name", "dosage", "shelf_life_months", "allowed_lot_prefixes", "description"]]
                st.dataframe(m_df, use_container_width=True)
            st.caption("ระบบใช้ตารางนี้ในการเทียบหาข้อผิดพลาดฉลากสลับรุ่น (Label Mix-up) และคำนวณ Shelf-Life ตามมาตรฐาน GMP")

    with c_inf2:
        with st.expander("💡 คำแนะนำการจัดแสง & สถาปัตยกรรมระดับอุตสาหกรรม (Optical & Architecture)", expanded=False):
            st.markdown("""
            * **ลดแสงสะท้อนบนซองฟอยล์ (Anti-Glare Optical Setup):**
              1. **Polarizing Filter:** ติดตั้ง CPL / Linear Polarizer หน้าเลนส์กล้อง
              2. **Diffused / Coaxial Light:** ใช้ไฟวงแหวนแบบมีแผ่นกระจายแสง หรือไฟ Coaxial ตัดเงาสะท้อนตรงจุดซีลฟอยล์
              3. **CLAHE:** ประมวลผลภาพด้วย Contrast-Limited Adaptive Histogram Equalization เพื่อดึงขอบตัวอักษรแม้จุดสะท้อนแสงจ้า
            * **ระบบสถาปัตยกรรมแบบแยกส่วน (Decoupled Production Architecture):**
              - รัน **`core_engine.py`** เพื่อประมวลผลกล้อง Real-time แยกเป็น Standalone Worker
              - ใช้งาน **`api_server.py` (FastAPI + WebSockets)** ส่งข้อมูลและสัญญาณ Alert ไปยัง SCADA/PLC
              - ฐานข้อมูล SQLite เปิดใช้งานโหมด **WAL (Write-Ahead Logging)** รองรับการอ่านและเขียนข้อมูลพร้อมกันแบบ Zero-lock
            """)
else:
    st.info("No inspection records found yet. Run an inspection to populate the historical log.")

# Sidebar cleanup button
if st.sidebar.button("🗑️ Reset Inspection Logs"):
    db.clear_history()
    st.session_state.last_inspection = None
    st.sidebar.success("Database cleared!")
    st.rerun()
