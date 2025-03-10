# Install required libraries
# !pip install torch transformers diffusers

# Import necessary modules
from diffusers import StableDiffusionPipeline
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")

# Example usage
prompt = "A futuristic cityscape at sunset"
print(f"Generating image for prompt: '{prompt}'")

try:
    # Generate the image
    image = pipe(prompt).images[0]
    print("Image generated successfully.")
    
    # Display the image in Colab
    from IPython.display import display
    display(image)
except Exception as e:
    print(f"Failed to generate image: {e}")
