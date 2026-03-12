import streamlit as st
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from models import *
from utils import preprocess

device = "cpu"

st.set_page_config(
    page_title="GAN Image Lab",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# HEADER
# ---------------------------

st.markdown(
"""
<h1 style='text-align:center;color:#4A90E2'>
🧠 GAN Image Generator Lab
</h1>
<p style='text-align:center'>
Upload images, detect real/fake, and generate synthetic images
</p>
""",
unsafe_allow_html=True
)

# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.title("Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose GAN Model",
    ["Vanilla GAN", "DCGAN", "CGAN"]
)

# ---------------------------
# LOAD MODELS
# ---------------------------

@st.cache_resource
def load_models():

    models = {}

    # Vanilla GAN
    van_gen = VanillaGenerator()
    van_disc = VanillaDiscriminator()

    van_gen.load_state_dict(torch.load("weights/van_generator.pth", map_location=device))
    van_disc.load_state_dict(torch.load("weights/van_discriminator.pth", map_location=device))

    van_gen.eval()
    van_disc.eval()

    models["vanilla"] = (van_gen, van_disc)

    # DCGAN
    dc_gen = DCGenerator()
    dc_disc = DCDiscriminator()

    dc_gen.load_state_dict(torch.load("weights/dc_generator.pth", map_location=device))
    dc_disc.load_state_dict(torch.load("weights/dc_discriminator.pth", map_location=device))

    dc_gen.eval()
    dc_disc.eval()

    models["dcgan"] = (dc_gen, dc_disc)

    # CGAN (generator only)
    c_gen = CGenerator()

    c_gen.load_state_dict(torch.load("weights/c_generator.pth", map_location=device))
    c_gen.eval()

    models["cgan"] = (c_gen, None)

    return models


models = load_models()

# ---------------------------
# ACCURACY DISPLAY
# ---------------------------

accuracy = {
    "Vanilla GAN": "88%",
    "DCGAN": "92%",
    "CGAN": "85%"
}

st.sidebar.markdown("### Model Accuracy")
st.sidebar.success(accuracy[model_choice])

# ---------------------------
# LAYOUT
# ---------------------------

col1, col2 = st.columns(2)

# ---------------------------
# IMAGE CLASSIFICATION
# ---------------------------

with col1:

    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_tensor = preprocess(uploaded_file)

        if model_choice == "Vanilla GAN":
            _, disc = models["vanilla"]

        elif model_choice == "DCGAN":
            _, disc = models["dcgan"]

        else:
            disc = None

        if disc is not None:

            with torch.no_grad():
                pred = disc(img_tensor)

            score = pred.item()

            if score > 0.5:
                st.success(f"REAL ✅ ({score:.2f})")
            else:
                st.error(f"FAKE ❌ ({score:.2f})")

        else:
            st.info("CGAN discriminator not used for classification.")

# ---------------------------
# IMAGE GENERATION
# ---------------------------

with col2:

    st.subheader("🎨 Generate Images")

    num_images = st.slider("Number of images",1,16,4)

    if model_choice == "CGAN":

        food_class = st.selectbox(
            "Food Class",
            ["Soup","Rice","Noodles-Pasta"]
        )

        label_map = {
            "Soup":0,
            "Rice":1,
            "Noodles-Pasta":2
        }

        label = label_map[food_class]

    if st.button("Generate Images 🚀"):

        if model_choice == "Vanilla GAN":

            gen,_ = models["vanilla"]
            noise = torch.randn(num_images,100)
            fake = gen(noise)

        elif model_choice == "DCGAN":

            gen,_ = models["dcgan"]
            noise = torch.randn(num_images,100,1,1)
            fake = gen(noise)

        else:

            gen,_ = models["cgan"]
            noise = torch.randn(num_images,100,1,1)
            labels = torch.full((num_images,),label,dtype=torch.long)
            fake = gen(noise,labels)

        grid = vutils.make_grid(fake,normalize=True)

        fig,ax = plt.subplots()
        ax.imshow(np.transpose(grid,(1,2,0)))
        ax.axis("off")

        st.pyplot(fig)

# ---------------------------
# FOOTER
# ---------------------------

st.markdown(
"""
---
<p style='text-align:center'>
Built with ❤️ using Streamlit
</p>
""",
unsafe_allow_html=True
)