import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.set_page_config(page_title="Dígitos — Câmera", page_icon="🔢")
st.title("🔢 Reconhecimento de Dígitos pela Câmera")

@st.cache_resource
def carregar_modelo():
    try:
        return tf.keras.models.load_model("cnn2d_mnist_digit_model.h5", compile=False)
    except TypeError:
        from tensorflow.keras.layers import Dense as _Dense
        class _DenseCompat(_Dense):
            def __init__(self, *a, quantization_config=None, **kw):
                super().__init__(*a, **kw)
        return tf.keras.models.load_model(
            "cnn2d_mnist_digit_model.h5",
            custom_objects={"Dense": _DenseCompat},
            compile=False,
        )

modelo = carregar_modelo()

def prever(imagem_pil):
    img = imagem_pil.convert("L")       # escala de cinza
    img = ImageOps.invert(img)          # fundo preto, dígito branco (MNIST)
    img = img.resize((28, 28))
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = x.reshape(1, 28, 28, 1)
    prob   = modelo.predict(x, verbose=0)[0]
    digito = int(np.argmax(prob))
    return digito, prob


foto = st.camera_input("Aponte a câmera para um dígito e tire a foto")

if foto:
    img = Image.open(foto)
    digito, prob = prever(img)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(foto, caption="Captura", use_container_width=True)
    with col2:
        st.metric("Dígito reconhecido", str(digito))
        st.metric("Confiança", f"{prob[digito]*100:.1f}%")
        st.markdown("**Todas as probabilidades:**")
        for i, p in enumerate(prob):
            st.progress(float(p), text=f"{i}: {p*100:.1f}%")
