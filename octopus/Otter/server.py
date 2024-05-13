import os, json
import base64
import torch
import transformers

from src.otter_ai.models.otter.modeling_otter import OtterForConditionalGeneration
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

IMAGE_PROCESSOR = transformers.CLIPImageProcessor()
PRECESSION = {"torch_dtype": torch.float32}
MODEL = OtterForConditionalGeneration.from_pretrained("checkpoints/Otter_MPT7B_EAI_2024-03-07-15-39-03", device_map="auto", **PRECESSION)
TENSOR_DTYPE =  torch.float32

app = Flask(__name__)

def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"

@app.route('/app/otter', methods=['POST'])
def process_request():
    # Extract the JSON content from the request
    data = request.json
    images = data['content']['images']
    prompt = data['content']['prompt']
    decoded_images = []
    for img in images.values():
        decoded_images.append(decode_image_from_base64(img))
  
    if isinstance(decoded_images, list):
        vision_x = IMAGE_PROCESSOR.preprocess(decoded_images, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")
    
    lang_x = MODEL.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )
    bad_words_id = MODEL.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    generated_text = MODEL.generate(
        vision_x=vision_x.to(MODEL.device, dtype = TENSOR_DTYPE),
        lang_x=lang_x["input_ids"].to(MODEL.device),
        attention_mask=lang_x["attention_mask"].to(MODEL.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id,
    )
    parsed_output = (
        MODEL.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )

    
    response_data = {
        "message": "Request processed successfully",
        "content": parsed_output
    }

    return jsonify(response_data)

def decode_image_from_base64(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    return image

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host = '172.21.25.95', port=5433)
