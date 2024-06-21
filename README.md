# Unconditional Image Generation

This project involved developing and training a diffusion model from scratch inspired by the architecture of Stable Diffusion. The model was trained using PyTorch framework. A diffusion probabilistic model is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. 

## Dataset 

The dataset used is the MNIST (digits) dataset. It consists of 70,000 labeled grayscale images of hand-written digits, each 28x28 pixels in size.
Some sample images from the dataset are shown : 

![image](https://github.com/Parth-Agarwal216/Image_Generation/assets/118837763/fa0c91de-ba5e-40bf-8ed9-b0e47fe12aca)

## Architecture

A modified U-Net architecture which comprises of multi-headed self attention, residual connections and time embeddings.

![U-NET](https://github.com/Parth-Agarwal216/Image_Generation/assets/118837763/f6d09ec8-b2f6-430e-9dce-faa48f63c963)

### Installation and Setup

1. **Install Dependencies:**
   Install the required libraries using:
```python
pip install -r requirements.txt
```

## Contributors
- [shubhranshu7](https://github.com/shubhranshu7)
- [Parth-Agarwal216](https://github.com/Parth-Agarwal216)
---
