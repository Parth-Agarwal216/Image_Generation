from diffusion_model import *

device = 'cpu'

def generate(
    diffusion_model,
    n_inference_steps=50,
):
    with torch.no_grad():

        sampler = DDPMSampler()
        sampler.set_inference_timesteps(n_inference_steps)

        image_shape = (1, 1, 28, 28)
        image = torch.randn(image_shape, device=device)

        diffusion_model.to(device)

        timesteps = sampler.timesteps
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(torch.tensor(timestep).view(1,)).to(device)
            model_input = image
            model_output = diffusion_model(model_input, time_embedding)
            image = sampler.step(timestep, image, model_output)
        image = image.permute(0, 2, 3, 1)
        image = image.detach().cpu()
        return image[0]

def gen_img():
    diffusion_model = Diffusion().to(device)
    checkpoint = torch.load("diffusion.tar", map_location=torch.device('cpu'))
    diffusion_model.load_state_dict(checkpoint)

    images = []
    for _ in range(12):
        image = generate(diffusion_model)
        image = image.squeeze()
        image = image.numpy()
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image, mode='L')
        images.append(image)

    return images
