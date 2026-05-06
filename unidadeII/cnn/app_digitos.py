import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


try:
    modelo = tf.keras.models.load_model("cnn2d_mnist_digit_model.keras", compile=False)
except TypeError:
    from tensorflow.keras.layers import Dense as _Dense
    class _DenseCompat(_Dense):
        def __init__(self, *a, quantization_config=None, **kw):
            super().__init__(*a, **kw)
    modelo = tf.keras.models.load_model(
        "cnn2d_mnist_digit_model.h5",
        custom_objects={"Dense": _DenseCompat},
        compile=False,
    )

st.title("Reconhecimento de Dígitos")

arquivo = st.file_uploader("Envie uma imagem do dígito (PNG/JPG)", type=["png", "jpg", "jpeg"])

if arquivo:
    img = Image.open(arquivo).convert("L")          # escala de cinza
    img = ImageOps.invert(img)                      # fundo preto, dígito branco (padrão MNIST)
    img = img.resize((28, 28))
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = x.reshape(1, 28, 28, 1)

    prob   = modelo.predict(x, verbose=0)[0]
    digito = int(np.argmax(prob))

    st.image(arquivo, width=200)
    st.markdown(f"## Dígito: **{digito}**")
    st.markdown(f"Confiança: `{prob[digito]*100:.1f}%`")
