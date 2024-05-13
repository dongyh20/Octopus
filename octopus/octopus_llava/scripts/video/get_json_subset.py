import json

with open("./data//LLaMA-VID-Finetune/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_fixdca30k_fixsynden40k_sg40kt2k_ori_with_video_chatgpt_maxtime_5min.json", "r") as f:
    data = json.load(f)

video_data = []
image_data = []
pure_data = []
for _ in data:
    
    if "image" in _ and len(image_data) < 8:
        image_data.append(_)

    if "video" in _ and len(video_data) < 8:
        # video_file = _['video']
        # suffix = video_file.split('.')[-1]
        # if suffix == 'pkl':
        #     import pdb; pdb.set_trace()
        video_data.append(_)

    if "image" not in _ and "video" not in _ and len(pure_data) < 8:
        pure_data.append(_)
    
    if len(image_data) == 8 and len(video_data) == 8 and len(pure_data) == 8:
        break

# import pdb; pdb.set_trace()
image_data += video_data
image_data += pure_data

with open("./data//LLaMA-VID-Finetune/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_fixdca30k_fixsynden40k_sg40kt2k_ori_with_video_chatgpt_maxtime_5min_subset.json", "w") as f:
    json.dump(image_data,f)
