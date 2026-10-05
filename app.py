from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import predict


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "floodframe"
    / "best_model.pth"
)


st.set_page_config(
    page_title="FloodFrame",
    page_icon="🌊",
    layout="centered",
)


st.title(
    "🌊 FloodFrame"
)

st.caption(
    "AI-assisted flood indication screening "
    "from camera frames."
)


st.markdown(
    """
FloodFrame membantu melakukan screening awal
terhadap frame kamera untuk mendeteksi indikasi
genangan atau banjir.

**AI hanya memberikan indikasi awal.**
Hasil tetap perlu diverifikasi oleh operator.
"""
)


uploaded_file = st.file_uploader(
    "Upload frame CCTV / gambar jalan",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Input Frame",
        use_container_width=True,
    )

    if not MODEL_PATH.exists():

        st.error(
            "Model belum tersedia. "
            "Download hasil training Kaggle "
            "dan letakkan pada "
            "runs/floodframe/best_model.pth"
        )

        st.stop()

    temp_path = (
        PROJECT_ROOT
        / "temp_prediction.jpg"
    )

    image.save(temp_path)

    with st.spinner(
        "Menganalisis frame..."
    ):

        (
            prediction,
            confidence,
            probabilities,
            classes,
        ) = predict(
            temp_path,
            MODEL_PATH,
        )

    st.divider()

    if prediction.lower() == "flood":

        st.error(
            "⚠️ Indikasi genangan / banjir terdeteksi"
        )

    else:

        st.success(
            "✅ Tidak ada indikasi banjir"
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prediction",
            prediction.upper(),
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2%}",
        )

    st.subheader(
        "Class Probability"
    )

    for name, probability in zip(
        classes,
        probabilities,
    ):

        st.write(
            f"**{name}**"
        )

        st.progress(
            float(probability)
        )

        st.caption(
            f"{probability:.2%}"
        )

    st.warning(
        "Confidence menunjukkan keyakinan model "
        "terhadap klasifikasi, bukan tingkat "
        "kedalaman atau severity banjir."
    )

    temp_path.unlink(
        missing_ok=True
    )