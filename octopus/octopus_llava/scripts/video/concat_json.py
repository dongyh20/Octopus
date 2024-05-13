import json
import os

with open("./data/LLaMA-VID-Finetune/llava_v1_5_mix665k_with_video_chatgpt_maxtime_5min.json", "r") as f:
    llama_vid = json.load(f)


# with open("/mnt/bn/vl-research/workspace/project/2023/LLaVA/playground/data/llava_instruct/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_fixdca30k_fixsynden40k_sg40kt2k_ori.json", "r") as f:
#     llava_1_6 = json.load(f)

video = []
counter = 0
for _ in llama_vid:
    if "prompt" in _:
        # import pdb; pdb.set_trace()
        video.append(_)
        if not os.path.exists(f"./data/LLaMA-VID-Finetune/{_['video']}"):
            print(f"File {_['video']} not exists")
            counter += 1

# for _ in llava_1_6:
#     if "image" in _:
#         # import pdb; pdb.set_trace()
#         if not os.path.exists(f"./data/LLaMA-VID-Finetune/{_['image']}"):
#             print(f"File {_['image']} not exists")
#             counter += 1

print(counter)

# final_json = llava_1_6 + video

# with open("./data/LLaMA-VID-Finetune/llava_158k_detailv3_reinstall_gpt4v24k_wild15k_mixdocvqa_fixdca30k_fixsynden40k_sg40kt2k_ori_with_video_chatgpt_maxtime_5min.json", "w") as f:
#     json.dump(final_json,f)
