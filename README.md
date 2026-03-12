# 🧠 GAN Image Generator Lab

This project is an interactive Streamlit application that lets you explore and play with different Generative Adversarial Network (GAN) models. You can test if images are "Real" or "Fake", and generate synthetic images on the fly!

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python installed (preferably version 3.8+).

### 1. Clone the Repository
```bash
git clone https://github.com/Prems1101/GAN_Image_Generator_Lab.git
cd GAN_Image_Generator_Lab
```

### 2. Install Dependencies
You need the following Python libraries:
* `streamlit`
* `torch` (PyTorch)
* `torchvision`
* `matplotlib`
* `Pillow`
* `numpy`

You can install them directly via pip:
```bash
pip install streamlit torch torchvision matplotlib Pillow numpy
```

### 3. Run the App
Launch the Streamlit interface with:
```bash
streamlit run app.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

---

## 🏗️ Model Architectures

The lab features three distinct GAN architectures, each designed for different educational and performance focuses. The networks generate/discriminate images at a resolution of **64x64**.

### 1. Vanilla GAN (`vanilla`)
The most basic form of a Generative Adversarial Network utilizing standard Multi-Layer Perceptrons (Linear/Dense Layers).
* **Generator:**
  * Uses sequential `nn.Linear` layers with `ReLU` activations.
  * Takes a latent space noise vector (`z_dim=100`) and upscales it through 256 -> 512 -> 1024 dimension hidden layers.
  * Outputs a flattened image corresponding to `3*64*64` using a `Tanh` activation function. 
* **Discriminator:**
  * Uses sequential `nn.Linear` layers with `LeakyReLU` activations.
  * Takes the flattened `3*64*64` representation and downscales it through 1024 -> 512 -> 256.
  * Outputs a continuous probability value between 0 and 1 via a `Sigmoid` activation.

### 2. Deep Convolutional GAN (`DCGAN`)
A more sophisticated architecture that utilizes Convolutional Neural Networks, proving much better at capturing spatial relationships in visual data.
* **Generator:**
  * Utlizes Transposed Convolutions (`nn.ConvTranspose2d`) to iteratively upscale the image from the initial (`100x1x1`) noise map.
  * Uses `BatchNorm2d` and `ReLU` at each upsampling block.
  * Outputs a stable `3x64x64` generated color image using a `Tanh` activation.
* **Discriminator:**
  * Acts as a standard Convolutional Neural Network image classifier.
  * Replaces pooling layers with strided convolutions (`nn.Conv2d` with stride=2) and utilizes `LeakyReLU` for downsampling.
  * Outputs a continuous probability via a `Sigmoid` activation.

### 3. Conditional GAN (`CGAN`)
An extension of the DCGAN that learns to map generated images to specified classes. In this lab, it generates images matching three specific food classes: `Soup`, `Rice`, and `Noodles-Pasta`.
* **Generator:**
  * A DCGAN-style architecture but with a structural twist: It accepts both the random noise AND an embedded class label (`nn.Embedding`).
  * The label and noise maps are concatenated together (`torch.cat`) before being passed through the transposed convolutional layers.
* **Discriminator** *(Backend Architecture shown, though not utilized in the UI)*:
  * Accepts both the image and a similarly embedded, spatially replicated target label.
  * Evaluates not just if the image looks real, but if it looks real *for that specific class*.

---

## 📁 Project Structure

* `app.py`: The Main Streamlit application and UI logic.
* `models.py`: PyTorch Module class definitions for the Generators and Discriminators for all three GAN variants.
* `utils.py`: Image loading and tensor preprocessing scripts for classification tasks.
* `weights/`: PyTorch weights files (`.pth`) for the pre-trained networks.
