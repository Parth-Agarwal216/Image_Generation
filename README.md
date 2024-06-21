# Unconditional Image Generation

This project involved developing and training a diffusion model from scratch inspired by the architecture of Stable Diffusion. The model was trained using PyTorch framework.

## Diffusion Model

A diffusion probabilistic model is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. 

## Architecture

This diffusion model uses a modified U-NET architecture which incorporates residual connections and multi headed self attention between the convolutional layers. Additionally, sinusoidal position embeddings are used to encode timestep t. This makes the neural network “know” at which specific time step (noise level) it is operating, for each image in a batch.

<div align="center">
  <img src="https://github.com/Parth-Agarwal216/Image_Generation/assets/118837763/23acc50f-a21c-4ef0-a075-b327747fcbc9" alt="image" width="500" />
</div>

## Dataset 

The dataset used is the MNIST (digits) dataset. It consists of 70,000 labeled grayscale images of hand-written digits, each 28x28 pixels in size.
Some sample images from the dataset are shown : 

<div align="center">
  <img src="https://github.com/Parth-Agarwal216/Image_Generation/assets/118837763/fa0c91de-ba5e-40bf-8ed9-b0e47fe12aca" alt="image" width="300" />
</div>

## The diffusion process

<div align="center">
  <img src="https://github.com/Parth-Agarwal216/Image_Generation/assets/118837763/abbac076-63bc-4bfc-a2df-a296370ccdfe" alt="image" width="500" />
</div>

### Installation and Setup

1. **Install Dependencies:**
   Install the required libraries using:
```python
pip install -r requirements.txt
```

2. **Run the Application:**
Execute the following command to run the app:
```python
streamlit run app.py
```

## Contributors
- [shubhranshu7](https://github.com/shubhranshu7)
- [Parth-Agarwal216](https://github.com/Parth-Agarwal216)
---
