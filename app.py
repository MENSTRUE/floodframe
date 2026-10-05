from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import load_model, predict_image


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "floodframe_efficientnet_b0.pth"
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


@st.cache_resource
def get_model():

    return load_model(
        MODEL_PATH
    )


if not MODEL_PATH.exists():

    st.error(
        "Model belum tersedia.\n\n"
        "Pastikan file berikut ada:\n\n"
        "`models/floodframe_efficientnet_b0.pth`"
    )

    st.stop()


try:

    model_data = get_model()

except Exception as error:

    st.error(
        "Model gagal dimuat."
    )

    st.exception(
        error
    )

    st.stop()


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

    with st.spinner(
        "Menganalisis frame..."
    ):

        (
            prediction,
            confidence,
            probabilities,
            classes,
        ) = predict_image(
            image=image,
            model_data=model_data,
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
            f"**{name.upper()}**"
        )

        st.progress(
            float(probability)
        )

        st.caption(
            f"{probability:.2%}"
        )

    st.warning(
        "Confidence menunjukkan tingkat keyakinan "
        "model terhadap klasifikasi, bukan tingkat "
        "keparahan atau kedalaman banjir."
    )