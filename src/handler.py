#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''

import runpod
import subprocess
import os


# Load models into VRAM here so they can be warm between requests


def huggingface_login():
    try:
        # Get the value of the TOKEN environment variable
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

        if token:
            # Run the huggingface-cli login command with the TOKEN environment variable
            subprocess.run(["huggingface-cli", "login",
                           "--token", token], check=True)

            # If the command was successful, you can print a success message or perform other actions
            print("Hugging Face login successful!")

        else:
            # Handle the case where the TOKEN environment variable is not set
            print(
                "TOKEN environment variable is not set. Please set it before running the command.")

    except subprocess.CalledProcessError as e:
        # If the command failed, you can print an error message or handle the error as needed
        error_message = f"Error running huggingface-cli login: {e}"
        print(error_message)


def install_oneflow():
    try:
        # Define the pip install command
        install_command = [
            "pip",
            "install",
            "--pre",
            "oneflow",
            "-f",
            "https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu117"
        ]

        # Execute the pip install command
        subprocess.run(install_command, check=True)

        # Installation successful
        print("OneFlow installed successfully.")
    except subprocess.CalledProcessError as e:
        # Error occurred during installation
        print(f"Error installing OneFlow: {e}")
    except Exception as e:
        # Other unexpected errors
        print(f"An error occurred: {e}")


def handler(job):
    job_input = job["input"]
    print(job_input)

    job_output = {}

    # Install OneFlow before using it
    install_oneflow()

    try:
        import torch
        # import oneflow as flow
        from pathlib import Path
        from onediff.infer_compiler import oneflow_compile
        from diffusers import StableDiffusionXLPipeline
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

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
        job_output["output"] = file_name
        return job_output
    except Exception as e:
        print(f"An error occurred: {e}")
        # do the things
        # return the output that you want to be returned like pre-signed URLs to output artifacts
        return {"output": "output"}


runpod.serverless.start({"handler": handler})
