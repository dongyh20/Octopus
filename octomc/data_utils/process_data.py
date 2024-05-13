from tqdm import tqdm
import os
import json
import shutil
path="./data"
tasks=os.listdir(path)

def get_biome(tasks):
    result={}
    for task in tqdm(tasks):
        tar_path=os.path.join(path,task,"subtask_1","observe.json")
        with open(tar_path,"r")as f:
            data=json.load(f)
            biome=data[0][1]['status']['biome']
            result[task]=biome
    with open("./biome.json","w")as f:
        f.write(json.dumps(result))

def rename(): #just need to do it once!
    for task in tqdm(sorted(tasks)):
        tar_path=os.path.join(path,task)
        for subtask in sorted(os.listdir(tar_path)):
            subtask_img=[]
            subtask_path=os.path.join(tar_path,subtask)
            for file in os.listdir(subtask_path):
                if file.endswith('png'):
                    subtask_img.append(os.path.join(subtask_path,file))
            for index,png in enumerate(sorted(subtask_img)):
                if "pic" in png:
                    continue
                else:
                    renamed_path=os.path.join(subtask_path,f"pic{index+1}.png")
                    shutil.move(png,renamed_path)   
def modified_plan():
    with open("data_utils/old_plan.json","r")as f:
        json_plan=json.load(f)
        new_plan={}
        for task in sorted(list(json_plan.keys())):
            planlist=json_plan[task]
            modified_planlist=[]
            for index,plan in enumerate(planlist):
                replaced_plan=plan.replace("1)",f"{index+1})")
                modified_planlist.append(replaced_plan)
            new_plan[task]=modified_planlist
    if not os.path.exists("data_utils/plan.json"):
        with open("data_utils/plan.json","w")as f:
            f.write(json.dumps(new_plan))    

def format_answer(answer):
    answer += "Explain:\n"
    answer += explain + "\n"
    answer += "Subtask:\n"
    answer += "\n".join(subtask_plan) + "\n"
    answer += "Code:\n"
    answer += code
    return answer

if __name__=="__main__":
    result={"meta":{"version":"v1","author":"LiuShuai","data":"2024-3-3"},"data":{}}
    mc_data={}
    with open("data_utils/plan.json","r")as f:
        plan=json.load(f)  
    for task in tqdm(sorted(tasks)):
        tar_path=os.path.join(path,task)
        for subtask in sorted(os.listdir(tar_path)):

            mc_data[f"{task}_{subtask}"]={}
            image_sequence=[] #the img sequence of each subtask
            relation_image_id=[]
            count_subtask=int(subtask.split("_")[1]) # the subtask k

            subtask_path=os.path.join(tar_path,subtask)
            # instruction
            with open(os.path.join(subtask_path,"input.json"),"r")as f:
                input=json.load(f)
                obs_obj=input['input'].split("Task Goal:")[0]
                instruction="Task Goal:"+input['input'].split("Task Goal:")[1]
            # response
            with open(os.path.join(subtask_path,"response.json"),"r")as f:
                response=json.load(f)
                answer=''
                explain=response['response']['explain']
                code=response['response']['code']
                subtask_plan=plan[task]
                answer=format_answer(answer)
            # reward
            with open(os.path.join(subtask_path,"feedback.json"),"r")as f:
                reward=json.load(f)
                main_reward=int(reward['main_succeed'])
            # image_id
            for png_file in os.listdir(subtask_path):
                if png_file.endswith("png"):
                    image_sequence.append(os.path.join(subtask_path,png_file))
            #rel ins id
            for cnt in range(1,count_subtask):
                relation_image_id.append(f"{task}_subtask_{cnt}")

            mc_data[f"{task}_{subtask}"]['objects']=obs_obj
            mc_data[f"{task}_{subtask}"]['instruction']=instruction
            mc_data[f"{task}_{subtask}"]['answer']=answer
            mc_data[f"{task}_{subtask}"]['image_ids']=sorted(image_sequence)
            mc_data[f"{task}_{subtask}"]['rel_ins_ids']=sorted(relation_image_id)        
            mc_data[f"{task}_{subtask}"]['reward']=1
            mc_data[f"{task}_{subtask}"]['main_reward']=main_reward
    print(mc_data)
    with open("data_utils/OctoMC.json","w")as f:
            f.write(json.dumps(mc_data))   