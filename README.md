# GenAI-Lab-3
# Variational Autoencoder (VAE) Implementation on MNIST

## 📌 Overview
This project implements a **Variational Autoencoder (VAE)** using PyTorch to generate handwritten digits from the MNIST dataset. Unlike standard autoencoders, this VAE learns a continuous latent space, allowing for the generation of **new, unseen data points** (generative modeling).

This lab was completed as part of **CSET419: Introduction to Generative AI**.

## 🧠 Key Concepts
- **Encoder:** Compresses input images (784 dimensions) into a latent probability distribution ($\mu, \sigma$).
- **Reparameterization Trick:** Enables backpropagation through random sampling ($z = \mu + \sigma \cdot \epsilon$).
- **Decoder:** Reconstructs images from the latent vector $z$.
- **Loss Function:** Combines **Reconstruction Loss** (BCE) + **KL Divergence** (Regularization).

## Results

### 1. Reconstruction Quality
The VAE successfully reconstructs inputs. Note the slight blurriness, which is characteristic of the probabilistic loss function used in VAEs (compared to MSE or GANs).
![Reconstruction](images/reconstruction.png)

### 2. Generative Capabilities
By sampling random noise from a standard normal distribution and feeding it to the decoder, the model generates new, legible digits.
![Generation](images/generation.png)

### 3. Latent Space Visualization
Mapping the test set to the 2D latent space reveals distinct clusters for different digits (e.g., 0s and 1s are far apart). The **KL Divergence** forces these clusters to be centered around (0,0).
![Latent Space](images/latent_space.png)

---

## Experiment: Removing KL Divergence
I performed an ablation study by training the model **without KL Divergence loss**. This effectively turns the VAE into a standard Autoencoder.

**Observations:**
* **Latent Space:** Without KL loss, the latent space "explodes" and clusters move far apart.
* **Generation:** Sampling from a standard normal distribution fails because the model did not learn to map the center (0,0) to meaningful data.

![Latent Space Comparison](images/latent_space_no_kl.png)

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/VAE-MNIST-Lab.git](https://github.com/your-username/VAE-MNIST-Lab.git)
