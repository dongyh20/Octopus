from flask import Flask, request, jsonify
from otter_ai import OtterForConditionalGeneration
import transformers
import io
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import base64
import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
from tqdm import tqdm
from datetime import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = Flask(__name__)
TOKEN = "sk-ICLRBESTPAPERAWARDOCTOPUS"

random_seed = 10043

torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def patch_images(frames):
    patch_resize_transform = Compose([Resize((224, 224)), ToTensor()])
    patch_images = torch.empty(0)

    for frame in frames:
        cur_patch_image = patch_resize_transform(frame).unsqueeze(0)
        patch_images = torch.cat((patch_images, cur_patch_image))

    patch_images = patch_images

    return patch_images

def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"

def get_response(encoded_frames, prompt, model=None, image_processor=None):
    vision_x = encoded_frames.unsqueeze(0).unsqueeze(0)

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
        max_new_tokens=2048,
        # temperature=1.2,
        num_beams=3,
        no_repeat_ngram_size=3,
        do_sample=True,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output


# loading models
device_id = "cuda:0"
model = OtterForConditionalGeneration.from_pretrained(
    "./model/Otter_MPT7B_EAI_GTA_1010_SFT/", device_map={"": device_id}, torch_dtype=torch.bfloat16,
)
model.text_tokenizer.padding_side = "left"
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()


@app.route('/app/otter', methods=['POST'])
def chat_gpt():
    start_time = datetime.now()
    data = request.get_json()
    print(f'{datetime.now() - start_time}: Data received', flush=True)
    token = data['token']
    if not token or token != f"{TOKEN}":
        return jsonify({"error": "Invalid token"}), 401

    content = data['content'][0] # prompt, images

    # processing images
    frames = []
    for image_id in content['images'].keys():
        image_base64 = content['images'][image_id]
        image_binary = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_binary))
        frames.append(image)
    encoded_frames = patch_images(frames)
    print(f'{datetime.now() - start_time}: Decoding received', flush=True)

    # processing prompts
    prompt = content['prompt']

    # get response
    print(f'{datetime.now() - start_time}: Start running model', flush=True)
    response = get_response(encoded_frames, prompt, model, image_processor)
    print(f'{datetime.now() - start_time}: Model run complete', flush=True)
    print(response)
    return jsonify({"result": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5433)
