import streamlit as st
import numpy as np
from PIL import Image


# TODO: importe e carregue o seu modelo aqui
# Exemplos:
#   import tensorflow as tf
#   model = tf.keras.models.load_model("meu_modelo.h5")


# TODO: defina os nomes das suas classes
CLASSES = ["classe_0", "classe_1", "classe_2"]


# TODO: ajuste o tamanho de entrada do seu modelo
INPUT_SIZE = (224, 224)   # (largura, altura)


def preprocessar(imagem: Image.Image) -> np.ndarray:
    """
    Pré-processa a imagem antes de enviá-la ao modelo.

    TODO: adapte conforme o pré-processamento exigido pelo seu modelo.
    """
    img = imagem.convert("RGB")                  # garante 3 canais
    img = img.resize(INPUT_SIZE)                 # redimensiona
    arr = np.array(img, dtype=np.float32) / 255.0  # normaliza [0, 1]
    arr = np.expand_dims(arr, axis=0)            # adiciona dimensão de batch
    return arr


def prever(imagem: Image.Image):
    """
    Executa a inferência e retorna o resultado.

    TODO: substitua o conteúdo desta função pela chamada ao seu modelo.
    """
    entrada = preprocessar(imagem)

    # ── exemplo ──────────────────────────────────────────────
    # probabilidades = model.predict(entrada)[0]
    # ─────────────────────────────────────────────────────────────────
    # Simulação para o template funcionar sem um modelo real:
    probabilidades = np.random.dirichlet(np.ones(len(CLASSES)))

    classe_pred = CLASSES[int(np.argmax(probabilidades))]
    confianca   = float(np.max(probabilidades))
    return classe_pred, confianca, probabilidades



#  Interface Streamlit

st.set_page_config(
    page_title="Classificador de Imagens",   # TODO: mude o título
    page_icon="🔍",                           # TODO: mude o ícone
    layout="centered",
)

st.title("🔍 Classificador de Imagens")       # TODO: mude o título
st.write(
    "Faça upload de uma imagem **ou** tire uma foto pela câmera "
    "para obter a predição do modelo."
)

#Seleção da fonte da imagem: Arquivo ou Câmera
st.divider()
fonte = st.radio(
    "Origem da imagem",
    options=["📁 Upload de arquivo", "📷 Câmera"],
    horizontal=True,
)

imagem_pil: Image.Image | None = None

if fonte == "📁 Upload de arquivo":
    arquivo = st.file_uploader(
        "Selecione uma imagem",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if arquivo is not None:
        imagem_pil = Image.open(arquivo)

else: 
    foto = st.camera_input("Tire uma foto")
    if foto is not None:
        imagem_pil = Image.open(foto)

#Exibição e predição
if imagem_pil is not None:
    st.divider()
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Imagem recebida")
        st.image(imagem_pil, use_container_width=True)

    with col_result:
        st.subheader("Resultado")
        with st.spinner("Processando..."):
            classe, confianca, probs = prever(imagem_pil)

        st.success(f"**Classe predita:** {classe}")
        st.metric("Confiança", f"{confianca * 100:.1f} %")

        st.divider()
        st.write("**Probabilidades por classe:**")
        for nome, prob in zip(CLASSES, probs):
            st.progress(
                float(prob),
                text=f"{nome}: {prob * 100:.1f} %",
            )
else:
    st.info("Aguardando imagem…")

#Rodapé
st.divider()
st.caption(
    "Template genérico — adapte `preprocessar()`, `prever()` e `CLASSES` "
    "para o seu modelo."
)
