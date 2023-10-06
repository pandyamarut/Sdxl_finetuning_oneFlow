import torch
# import oneflow as flow
from pathlib import Path
from onediff.infer_compiler import oneflow_compile
from diffusers import StableDiffusionXLPipeline


# Configuration
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0" # Please replace with your model path
prompt = "A little white dog playing on the sea floor, the sun shining in, swimming some beautiful Colorful goldfish, bubbles，by Yang J, pixiv contest winner, furry art, falling star on the background, bubbly underwater scenery, the cutest kitten ever, beautiful avatar pictures"
saved_image = "sdxl-base-out-dog.png"
file_name = "unetyyy_compiled"

pipe = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

pipe.unet = oneflow_compile(pipe.unet)
if Path(file_name).exists():
    print(f"Loading the compiled graph from {file_name}...")
    pipe.unet.warmup_with_load(file_name)


image = pipe(prompt).images[0]
print(f"Saved the image to {saved_image}.")
image.save(saved_image)

if not Path(file_name).exists():
    pipe.unet.save_graph(file_name)
    print(f"Saved the compiled graph to {file_name}.")
