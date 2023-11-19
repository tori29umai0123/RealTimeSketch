import gradio as gr
from PIL import Image
import cv2
import numpy as np
import sys
import webbrowser
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image
import torch

sd_model_path ="Models/SD"
lcm_lora_path = "Models/lcm_lora"
controlnet_canny_path ="Models/controlnet/canny"
controlnet_scribble_path ="Models/controlnet/scribble"

# パイプラインをグローバル変数として保持します
canny_pipe = None
scribble_pipe = None

low_threshold = 100
high_threshold = 200
img_width = 512
img_height = 768
init_img = Image.fromarray(np.ones((img_height, img_width, 3), dtype=np.uint8) * 255)
init_img_path = 'init.png'
init_img.save(init_img_path)

def Illust_generation_canny(np_img,  prompt: str, c_weight_input: float):
    controlnet = ControlNetModel.from_pretrained(controlnet_canny_path, torch_dtype=torch.float16)

    image = cv2.Canny(np_img, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    if np_img is None:
        return
    global canny_pipe
    if canny_pipe is None:
        scribble_pipe = None
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm_lora_path)
    generator = torch.Generator("cuda").manual_seed(1)

    image = pipe(
        prompt,
        image=canny_image,
        num_inference_steps=4,
        guidance_scale=1.5,
        controlnet_conditioning_scale=float(c_weight_input),
        cross_attention_kwargs={"scale": 1},
        generator=generator
        ).images[0]

    return image


def Illust_generation_scribble(np_img,  prompt: str, c_weight_input: float):
    controlnet = ControlNetModel.from_pretrained(controlnet_scribble_path, torch_dtype=torch.float16)

    image = cv2.Canny(np_img, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    if np_img is None:
        return
    global scribble_pipe
    if scribble_pipe is None:
        canny_pipe = None
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm_lora_path)
    generator = torch.Generator("cuda").manual_seed(1)

    image = pipe(
        prompt,
        image=canny_image,
        num_inference_steps=4,
        guidance_scale=1.5,
        controlnet_conditioning_scale=float(c_weight_input),
        cross_attention_kwargs={"scale": 1},
        generator=generator
        ).images[0]

    return image

def Mode_check(np_img,  prompt: str, c_weight_input: float, scribble_mode):
    if scribble_mode == True:
        image = Illust_generation_scribble(np_img,  prompt, c_weight_input)
        return image
    else:
        image = Illust_generation_canny(np_img,  prompt, c_weight_input)
        return image




with gr.Blocks() as ui:
    prompt_input = gr.Textbox(label="prompt", value="1girl")
    c_weight_input = gr.Slider(minimum=0, maximum=1.0, value=0.5, label="control weight")
    mode_check = gr.Checkbox(label="ScribbleMode", value=True)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                source="upload",
                tool="color-sketch",
                value=init_img_path,
                width=img_width,
                height=img_height,
                interactive=True,
            )
        with gr.Column():
            image_color_output = gr.Image(width=img_width, height=img_height)

    image_input.change(
        fn=Mode_check,
        inputs=[image_input, prompt_input, c_weight_input,mode_check],
        outputs=[image_color_output],
        show_progress='hidden'
    )
    prompt_input.change(
        fn=Mode_check,
        inputs=[image_input, prompt_input, c_weight_input,mode_check],
        outputs=[image_color_output],
        show_progress='hidden'
    )
    c_weight_input.change(
        fn=Mode_check,
        inputs=[image_input, prompt_input, c_weight_input,mode_check],
        outputs=[image_color_output],
        show_progress='hidden'
    )
    mode_check.change(
        fn=Mode_check,
        inputs=[image_input, prompt_input, c_weight_input,mode_check],
        outputs=[image_color_output],
        show_progress='hidden'
    )
ui.queue()
ui.launch()