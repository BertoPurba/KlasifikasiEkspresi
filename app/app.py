import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# =========================
# 1. Pengaturan halaman
# =========================

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="centered",
)


# =========================
# 2. Kelas & tampilan tiap ekspresi
# =========================
# Urutan class_names HARUS sama dengan urutan output model.

class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# label Indonesia, emoji, warna utama, warna latar lembut
EMOTIONS = {
    "angry":    ("Marah",     "😠", "#D64545", "#FCE9E7"),
    "disgust":  ("Jijik",     "🤢", "#6E9B3A", "#EEF4E2"),
    "fear":     ("Takut",     "😨", "#7B5EA7", "#EFE9F7"),
    "happy":    ("Senang",    "😄", "#E8A317", "#FDF3D9"),
    "neutral":  ("Netral",    "😐", "#6B7A8F", "#EAEEF3"),
    "sad":      ("Sedih",     "😢", "#3B7DD8", "#E4EEFB"),
    "surprise": ("Terkejut",  "😲", "#E2702B", "#FDEBDD"),
}


# =========================
# 3. Styling (CSS)
# =========================

THEMES = {
    "light": {
        "ink": "#1D2433", "muted": "#5F6B7F", "line": "#DDE3EC",
        "paper": "#F6F8FC", "card": "#FFFFFF", "accent": "#3B5BDB",
        "track": "#E6EBF3", "dash": "#B9C4D8",
    },
    "dark": {
        "ink": "#EDF1F8", "muted": "#9AA7BD", "line": "#2E384B",
        "paper": "#0F1420", "card": "#171E2E", "accent": "#7C93FF",
        "track": "#263047", "dash": "#3D4A66",
    },
}

st.session_state.setdefault("dark_mode", False)
is_dark = st.session_state["dark_mode"]
T = THEMES["dark" if is_dark else "light"]

theme_vars = "".join(f"--{k}: {v};" for k, v in T.items())
st.markdown(
    f"""<style>
:root {{ {theme_vars} color-scheme: {"dark" if is_dark else "light"}; }}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Figtree:wght@400;500;600&display=swap');

html, body, .stApp, [class*="css"] {
    font-family: 'Figtree', system-ui, sans-serif;
    color: var(--ink);
}
.stApp { background: var(--paper); }

/* Paksa warna teks mengikuti tema aplikasi (bukan tema bawaan Streamlit/sistem) */
.stApp p, .stApp span, .stApp label, .stApp li,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"], [data-testid="stSpinner"] {
    color: var(--ink);
}
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--ink); }
hr { border-color: var(--line) !important; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.block-container { max-width: 760px; padding-top: 2.4rem; padding-bottom: 4rem; }

h1, h2, h3 { font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.01em; color: var(--ink); }

/* Header */
.app-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800;
    font-size: 2.5rem;
    line-height: 1.1;
    margin: 0 0 .5rem 0;
}
.app-sub { color: var(--muted); font-size: 1.05rem; max-width: 34rem; margin-bottom: 1.6rem; }

/* Tabs */
div[data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); background: transparent; }
button[data-baseweb="tab"] {
    font-family: 'Figtree', sans-serif;
    font-weight: 600;
    background: transparent !important;
}
button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
    color: var(--muted) !important;
    font-weight: 600;
}
button[data-baseweb="tab"]:hover p { color: var(--ink) !important; }
button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span { color: var(--accent) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
div[data-baseweb="tab-border"] { background-color: var(--line) !important; }

/* Uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card);
    border: 2px dashed var(--dash);
    border-radius: 16px;
    padding: 1.6rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); }
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: var(--card);
    color: var(--ink);
    border: 1px solid var(--line);
}
[data-testid="stFileUploaderDropzone"] button * { color: var(--ink) !important; }
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }

/* Kamera */
[data-testid="stCameraInput"] button {
    background: var(--card);
    color: var(--ink);
    border: 1px solid var(--line);
}
[data-testid="stCameraInput"] button * { color: var(--ink) !important; }

/* Gambar */
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--line); }
[data-testid="stImageCaption"] { color: var(--muted); font-size: .85rem; }

/* Kartu hasil */
.result {
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    margin: .4rem 0 1.4rem 0;
    border: 1px solid rgba(0,0,0,.05);
}
.result .emoji { font-size: 4.2rem; line-height: 1; }
.result .name {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800;
    font-size: 2.3rem;
    line-height: 1.05;
}
.result .conf { font-size: 1.05rem; margin-top: .3rem; color: var(--ink); opacity: .75; }
.result .conf b { opacity: 1; }

/* Bar probabilitas */
.bars-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.15rem; margin: 1.2rem 0 .7rem; }
.bar-row { display: flex; align-items: center; gap: .8rem; margin-bottom: .55rem; }
.bar-label { width: 6.6rem; font-weight: 500; white-space: nowrap; }
.bar-track { flex: 1; height: 12px; background: var(--track); border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.bar-val { width: 3.8rem; text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.bar-row.top .bar-label, .bar-row.top .bar-val { color: var(--ink); font-weight: 700; }

/* Info kecil */
.note { color: var(--muted); font-size: .88rem; margin-top: 1.6rem; }

@media (max-width: 600px) {
    .app-title { font-size: 1.9rem; }
    .result { flex-direction: column; text-align: center; gap: .6rem; }
    .bar-label { width: 5.2rem; }
}
@media (prefers-reduced-motion: no-preference) {
    .bar-fill { transition: width .5s ease; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# 4. Load model & face detector
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/improved_cnn_best.keras")


@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


try:
    model = load_model()
except Exception as e:
    st.error(
        "Model tidak bisa dimuat. Pastikan file `models/improved_cnn_best.keras` ada "
        "di folder yang sama dengan app ini."
    )
    st.caption(f"Detail error: {e}")
    st.stop()

face_detector = load_face_detector()


# =========================
# 5. Fungsi bantu
# =========================

def detect_largest_face(rgb_array):
    """Return (box, jumlah_wajah). box = (x, y, w, h) atau None."""
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    if len(faces) == 0:
        return None, 0
    box = max(faces, key=lambda f: f[2] * f[3])
    return tuple(int(v) for v in box), len(faces)


def crop_face(rgb_array, box, pad_ratio=0.2):
    x, y, w, h = box
    pad = int(pad_ratio * max(w, h))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2 = min(rgb_array.shape[1], x + w + pad)
    y2 = min(rgb_array.shape[0], y + h + pad)
    return rgb_array[y1:y2, x1:x2]


def preprocess(face_rgb):
    img = Image.fromarray(face_rgb).convert("L").resize((48, 48))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)   # channel
    arr = np.expand_dims(arr, axis=0)    # batch
    return arr


def predict(face_rgb):
    probs = model.predict(preprocess(face_rgb), verbose=0)[0]
    return probs


def draw_box(rgb_array, box, color_hex):
    out = rgb_array.copy()
    x, y, w, h = box
    r, g, b = (int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    thickness = max(2, int(0.006 * max(out.shape[:2])))
    cv2.rectangle(out, (x, y), (x + w, y + h), (r, g, b), thickness)
    return out


def render_result(probs):
    idx = int(np.argmax(probs))
    name = class_names[idx]
    label, emoji, color, tint = EMOTIONS[name]
    if is_dark:
        tint = color + "2E"  # warna emosi transparan di atas latar gelap
    conf = float(probs[idx]) * 100

    st.markdown(
        f"""<div class="result" style="background:{tint}">
<div class="emoji">{emoji}</div>
<div>
<div class="name" style="color:{color}">{label}</div>
<div class="conf">Keyakinan model <b>{conf:.1f}%</b></div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Bar semua kelas, urut dari yang tertinggi
    order = np.argsort(probs)[::-1]
    rows = []
    for rank, i in enumerate(order):
        n = class_names[i]
        lab, emo, col, _ = EMOTIONS[n]
        pct = float(probs[i]) * 100
        top = " top" if rank == 0 else ""
        rows.append(
            f'<div class="bar-row{top}">'
            f'<div class="bar-label">{emo} {lab}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%;background:{col}"></div></div>'
            f'<div class="bar-val">{pct:.1f}%</div>'
            f"</div>"
        )
    st.markdown(
        '<div class="bars-title">Peluang tiap ekspresi</div>' + "".join(rows),
        unsafe_allow_html=True,
    )

    return color


def analyze(pil_image):
    rgb = np.array(pil_image.convert("RGB"))

    box, n_faces = detect_largest_face(rgb)

    if box is None:
        st.image(pil_image, caption="Gambar yang dipakai", use_container_width=True)
        st.warning(
            "Wajah tidak terdeteksi. Coba foto dengan wajah menghadap depan, "
            "pencahayaan cukup, dan ukuran wajah cukup besar."
        )
        return

    face = crop_face(rgb, box)
    probs = predict(face)

    top_name = class_names[int(np.argmax(probs))]
    color = EMOTIONS[top_name][2]

    render_result(probs)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(draw_box(rgb, box, color), caption="Wajah yang terdeteksi", use_container_width=True)
    with col2:
        st.image(face, caption="Potongan wajah untuk model", use_container_width=True)

    if n_faces > 1:
        st.info(f"Terdeteksi {n_faces} wajah. Yang dianalisis adalah wajah terbesar.")


# =========================
# 6. Halaman utama
# =========================

head_l, head_r = st.columns([4, 1])
with head_r:
    st.toggle("🌙 Mode gelap", key="dark_mode")
with head_l:
    st.markdown('<div class="app-title">Facial Expression<br>Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Unggah foto atau ambil gambar dari kamera. '
    "Model akan mengenali wajah lalu menebak ekspresinya.</div>",
    unsafe_allow_html=True,
)

tab_upload, tab_camera = st.tabs(["Unggah foto", "Pakai kamera"])

image_source = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Pilih gambar (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file)

with tab_camera:
    camera_file = st.camera_input("Ambil foto", label_visibility="collapsed")
    if camera_file is not None:
        image_source = Image.open(camera_file)

if image_source is not None:
    st.divider()
    with st.spinner("Menganalisis ekspresi..."):
        analyze(image_source)
else:
    st.markdown(
        '<div class="note">Belum ada gambar. Unggah foto atau ambil dari kamera untuk mulai.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="note">Hasil ini prediksi model, bukan penilaian pasti tentang perasaan seseorang.</div>',
    unsafe_allow_html=True,
)