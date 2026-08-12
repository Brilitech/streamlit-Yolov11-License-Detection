# app.py
# Aplikasi Streamlit untuk Deteksi Plat Nomor Kendaraan di SMK Muhammudiayah Kudus
# Menggunakan YOLOv11 + EasyOCR

import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import easyocr
import torch
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# ------------------------------
# 1. Konfigurasi Halaman
# ------------------------------
st.set_page_config(
    page_title="Deteksi Plat Nomor - SMK Muhammadiyah Kudus",
    page_icon="🚗",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3c72;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #2a5298;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🚗 Deteksi Plat Nomor Kendaraan</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">SMK Muhammadiyah Kudus</div>', unsafe_allow_html=True)

# ------------------------------
# 2. Cache Model (dimuat sekali)
# ------------------------------
@st.cache_resource
def load_anpr(model_path="anpr_best.pt"):
    """Memuat model YOLO dan EasyOCR."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return model, reader, device

try:
    model, reader, device = load_anpr()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'anpr_best.pt' ada di direktori yang sama. Error: {e}")
    st.stop()

# ------------------------------
# 3. Fungsi Deteksi
# ------------------------------
def detect_plate(image_np):
    """Deteksi plat dan OCR pada gambar numpy (BGR)."""
    results = model.predict(image_np, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
    annotated = image_np.copy()
    ann = Annotator(annotated, line_width=4)
    plate_texts = []

    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox)
        roi = image_np[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = reader.readtext(gray, detail=0, paragraph=True)
        text = " ".join(text).strip()
        plate_texts.append(text)
        ann.box_label(bbox, label=text, color=colors(17, True))

    return annotated, plate_texts

# ------------------------------
# 4. Sidebar Pilihan Input
# ------------------------------
st.sidebar.header("📥 Sumber Input")
option = st.sidebar.radio(
    "Pilih metode input:",
    ["📷 Upload Gambar", "🎥 Upload Video", "📹 Webcam"]
)

# ------------------------------
# 5. Penanganan Input
# ------------------------------
if option == "📷 Upload Gambar":
    st.subheader("Upload Gambar")
    uploaded_file = st.file_uploader("Pilih file gambar (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("🔍 Mendeteksi plat nomor..."):
            annotated_bgr, texts = detect_plate(img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Gambar Asli", use_container_width=True)
        with col2:
            st.image(annotated_rgb, caption="Hasil Deteksi", use_container_width=True)

        if texts and any(texts):
            st.success(f"✅ Plat terdeteksi: {', '.join([t for t in texts if t])}")
        else:
            st.warning("⚠️ Tidak ada plat nomor yang terdeteksi.")

elif option == "🎥 Upload Video":
    st.subheader("Upload Video")
    uploaded_video = st.file_uploader("Pilih file video (mp4, avi, mov)", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Simpan video sementara
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Proses video
        with st.spinner("⏳ Memproses video, mohon tunggu..."):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Gagal membuka video.")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30

                # Output video
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                frame_count = 0
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated, _ = detect_plate(frame)
                    out.write(annotated)
                    frame_count += 1
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                out.release()
                progress_bar.empty()

                st.success("✅ Proses video selesai!")
                st.video(out_path)

                # Hapus file sementara
                os.unlink(video_path)
                os.unlink(out_path)

elif option == "📹 Webcam":
    st.subheader("Deteksi Real-time melalui Webcam")
    st.info("Klik 'Start' untuk mengaktifkan kamera. Deteksi plat akan berjalan secara langsung.")

    # Konfigurasi WebRTC
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, _ = detect_plate(img)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="webcam-detection",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ------------------------------
# 6. Footer
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Informasi Model**
    - YOLOv11 + EasyOCR
    - Model plat: `anpr_best.pt`
    - GPU: {} 
    """.format("✅ Aktif" if torch.cuda.is_available() else "❌ Tidak terdeteksi (CPU)")
)

st.markdown(
    """
    <hr>
    <div style='text-align: center; color: gray;'>
        Dibuat dengan ❤️ oleh <b>SMK Muhammadiyah Kudus</b> | Teknik Komputer dan Jaringan
    </div>
    """,
    unsafe_allow_html=True
)