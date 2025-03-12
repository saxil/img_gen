import torch
from diffusers import StableDiffusionPipeline

def generate_image_local(prompt, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    """
    Generates an image from a text prompt using a local Stable Diffusion model.

    Args:
        prompt (str): The text description of the desired image.
        model_id (str): The Stable Diffusion model ID (e.g., "runwayml/stable-diffusion-v1-5").
        device (str): The device to use ("cuda" for GPU, "cpu" for CPU).

    Returns:
        PIL.Image.Image: The generated image, or None if an error occurs.
    """
    try:
        # Load the Stable Diffusion pipeline
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)

        # Generate the image
        image = pipe(prompt).images[0]
        return image

    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def display_image(image):
    """Displays a PIL Image object."""
    if image:
        image.show() #or image.save("generated_local_image.png")

# Example usage:
prompt = "A majestic lion in a vibrant jungle, digital art"
model_id = "runwayml/stable-diffusion-v1-5" # or other stable diffusion models.
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

generated_image = generate_image_local(prompt, model_id, device)
display_image(generated_image)

if generated_image:
    generated_image.save("generated_local_image.png") #Saves the image.