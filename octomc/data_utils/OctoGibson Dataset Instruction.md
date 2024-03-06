## OctoGibson Dataset Instruction

We provide the OctoGibson training dataset to apply future research.

### OctoGibsonDataset_RawImages.zip

We provide the raw images in OctoGibsonDataset_RawImages.zip 

The dataset structure is as follows:

```
.
├── TASK_NAME
│   ├── subtask_1
│   │   ├── rgb0_detect_surroundings.png
│   │   ├── rgb1_detect_surroundings.png
│   │   ├── rgb2_detect_surroundings.png
│   │   ├── rgb3_detect_surroundings.png
│   │   ├── rgb4_detect_surroundings.png
│   │   ├── rgb5_detect_surroundings.png
│   │   ├── rgb6_detect_surroundings.png
│   │   ├── rgb7_detect_surroundings.png
│   │   ├── rgb8_BEV_surroundings.png
│   │   └── rgb9_BEV_surroundings.png
│   ├── subtask_2
│   │   ├── rgb0_detect_surroundings.png
│   │   ├── rgb1_detect_surroundings.png
│   │   ├── rgb2_detect_surroundings.png
│   │   ├── rgb3_detect_surroundings.png
│   │   ├── rgb4_detect_surroundings.png
│   │   ├── rgb5_detect_surroundings.png
│   │   ├── rgb6_detect_surroundings.png
│   │   ├── rgb7_detect_surroundings.png
│   │   ├── rgb8_BEV_surroundings.png
│   │   └── rgb9_BEV_surroundings.png
│   ├── subtask_3
...
```

Each task directory contains 14 subtask folders. All the images have the resolution of 224*224.

### OctoGibson.json

In OctoGibson.json, we provide the processed data, consisting of instructions, answers and code generated from GPT-4 and so on. 

Take `cook_a_bread_slice_fail_subtask_1` as an example:

```
"cook_a_bread_slice_fail_subtask_1": {
            "objects": "Observed Objects: ...",
            "relations": "Observed Relations: ...",
            "instruction": "Inventory:..., Task Goal:..., Previous Action Code:..., Execution error:...",
            "answer": "...",
            "image_ids": [
                "cook_a_bread_slice_fail_subtask_1_rgb5_detect_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb7_detect_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb2_detect_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb4_detect_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb9_BEV_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb1_detect_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb3_detect_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb6_detect_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb0_detect_surroundings",
                "cook_a_bread_slice_fail_subtask_1_rgb8_BEV_surroundings"
            ],
            "rel_ins_ids": [],
            "reward": 1,
            "main_reward": 0
```

- We collect the `object information` and `relations` from the input messages we create, demonstrating the objects' name, ability, and relations with others.
- For `instruction`, we collect Inventory, Task Goal, Previous Action Code, Execution Error from the input message.
- For `answer`, we simply record all the GPT-4 response.
- For `image_ids`, we collect all the images related to the corresponding subtask.
- For `reward`, we use 1 to represent that the corresponding subtask is executed correctly and 0 for the opposite. Anything causes the subtask fails leads to 0 for this subtask, including code error, subtask goal error and so on.
- For `main_reward`, if a task fails, all the main_reward in its subtask folders are 0, indicating that the corresponding plan cannot lead to a success. If a task succeeds, for each subtask that are correctly executed, we assign 1 to the main_task reward of this subtask, otherwise 0. We use this to illustrate whether a chain of action can lead to a success for a task. 

### OctoGibson_images.json

In OctoGibson_images.json, we provide the images encoded by base64, which derive from the original images with a resolution of 224*224.

