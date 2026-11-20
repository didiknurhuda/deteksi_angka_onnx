import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import onnxruntime as ort
# ... (import lainnya tetap sama, hapus import tensorflow) ...

st.set_page_config(page_title="Demo CNN Deteksi Angka MNIST", layout="wide")




@st.cache_resource
def load_models():
    return ort.InferenceSession("mnist_cnn.onnx")

session = load_models()
input_name = session.get_inputs()[0].name
# Di dalam blok pemrosesan gambar:
outputs = session.run(None, {input_name: tensor_input})

probabilities = outputs[0][0]
conv1_act = outputs[1]
conv2_act = outputs[2]
prediction = int(np.argmax(probabilities))


def preprocess_image(image_data):
    img = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    bbox = img.getbbox()
    if bbox is None:
        return None

    img_cropped = img.crop(bbox)
    max_side = max(img_cropped.size)
    padded_size = int(max_side * 1.4)

    img_square = Image.new("L", (padded_size, padded_size), 0)
    paste_x = (padded_size - img_cropped.width) // 2
    paste_y = (padded_size - img_cropped.height) // 2
    img_square.paste(img_cropped, (paste_x, paste_y))

    img_28x28 = img_square.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_28x28, dtype="float32") / 255.0

    # Bentuk tensor (1, 28, 28, 1) sesuai input CNN
    tensor_input = img_array.reshape(1, 28, 28, 1)
    return img_28x28, tensor_input


def plot_feature_maps(feature_map, n_cols, title):
    n_filters = feature_map.shape[-1]
    n_rows = int(np.ceil(n_filters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n_filters:
            axes[i].imshow(feature_map[0, :, :, i], cmap="viridis")
            axes[i].set_title(f"Filter {i+1}", fontsize=9)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


st.title("Visualisasi Convolutional Neural Network (CNN) - MNIST")
st.markdown(
    "Tuliskan angka **0–9** pada kanvas untuk melihat hasil klasifikasi sekaligus "
    "bagaimana setiap *filter* konvolusi merespons goresan tulisan tangan Anda."
)

model, activation_model = load_models()

col1, col2, col3 = st.columns([1.2, 1, 1.5])

with col1:
    st.subheader("1. Input Kanvas")
    stroke_width = st.slider("Ketebalan kuas", 10, 30, 18)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

if canvas_result.image_data is not None:
    processed = preprocess_image(canvas_result.image_data)

    if processed is not None:
        img_28x28, tensor_input = processed

        # Prediksi kelas dan ekstraksi aktivasi layer konvolusi
        probabilities = model.predict(tensor_input, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        conv1_act, conv2_act = activation_model.predict(tensor_input, verbose=0)

        with col2:
            st.subheader("2. Input Tensor (28×28×1)")
            st.image(
                img_28x28.resize((200, 200), Image.Resampling.NEAREST),
                caption="Citra setelah dipusatkan",
            )
            st.metric(
                label="Prediksi CNN",
                value=str(prediction),
                delta=f"Probabilitas: {probabilities[prediction]*100:.1f}%",
            )

        with col3:
            st.subheader("3. Probabilitas Output (Softmax)")
            prob_df = pd.DataFrame({
                "Angka": [str(i) for i in range(10)],
                "Probabilitas (%)": probabilities * 100,
            })
            st.bar_chart(prob_df.set_index("Angka"))

        st.divider()
        st.subheader("4. Visualisasi Aktivasi Layer Konvolusi (Feature Maps)")
        st.caption(
            "Warna kuning terang menunjukkan area dengan nilai aktivasi ReLU tinggi "
            "(fitur berhasil dideteksi oleh filter), sedangkan ungu gelap bernilai nol."
        )

        act_col1, act_col2 = st.columns(2)

        with act_col1:
            fig1 = plot_feature_maps(
                conv1_act,
                n_cols=4,
                title="Conv2D Layer 1 (8 Filter, Output 26×26)",
            )
            st.pyplot(fig1)
            plt.close(fig1)

        with act_col2:
            fig2 = plot_feature_maps(
                conv2_act,
                n_cols=4,
                title="Conv2D Layer 2 (16 Filter, Output 11×11)",
            )
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        with col2:
            st.info("Silakan tulis satu angka pada kanvas hitam di sebelah kiri.")