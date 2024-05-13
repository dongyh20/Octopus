import os
import json
import time
import yaml
import argparse
from imp import reload

import parse_json
import query as query
import utils.gpt_utils as gu 
from gpt_request import gpt_request



def parse_args():
    description = "EVLM_gpt_process"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('idx', type=int, help='Index for data entry')
    parser.add_argument('process_path', type=str, default="/PATH/TO/CONFIG", help='path for loading process config')
    return parser.parse_args()


def gpt_process(args):
    idx = args.idx
    process_config = args.process_path
    process_cfg = yaml.load(open(process_config, "r"), Loader=yaml.FullLoader)
    sub_num = process_cfg["process"]["sub_num"]
    log_path = process_cfg["process"]["log_path"]
    data_path = process_cfg["process"]["data_path"]
    task_path = process_cfg["process"]["task_path"]
    
    with open(task_path) as f:
        data = json.load(f)
        
    EVLM_name = sorted(list(data))[idx]
    if "train" in data[EVLM_name]['split']: # check if it is training data
        task_data = data[EVLM_name] # all configs for this task
        gpt_task_name = task_data['detailed_name'] # detailed task name for data collection with GPT-4
        save_path = gu.f_mkdir(os.path.join(data_path, gpt_task_name))
        
        # main task loop
        main_task_flag = False
        subtask_iter = 1
        gpt_query = query.Query()
        for i in range(1, sub_num): # max subtask numbers set to sub_num
            gu.f_mkdir(os.path.join(save_path, f"subtask_{i}")) # pre-build all the subtasks
        
        while True:
            # make the directory
            sub_save_path = gu.f_mkdir(os.path.join(save_path, f"subtask_{subtask_iter}"))
            
            # init pipeline for each subtask
            while True:
                data = gu.get_finished_task(log_path)
                if gpt_task_name in data.keys():
                    return 0
                if os.path.exists(os.path.join(sub_save_path, 'task1.json')):
                    break   
                time.sleep(1)
            while True:
                statinfo = os.stat(os.path.join(sub_save_path, 'task1.json')) 
                if statinfo.st_size > 0:
                    break
            human_info = parse_json.parse_json(path=os.path.join(sub_save_path, "task1.json"))
            
            data = gu.get_finished_task(log_path)
            if gpt_task_name in data.keys():
                return 0
            
            # subtask loop, when a subtask is finished, close the loop
            while True:
                system_message = gpt_query.render_system_message()
                human_message = gpt_query.render_human_message(human_info)
                content = system_message.content + "\n\n" + human_message.content
                gu.save_input(sub_save_path, human_message.content)
                print("start query")
                
                # query process
                succuss = True
                while succuss:
                    try:
                        response = gpt_request(content)
                        succuss = False
                    except Exception as e:
                        print(f"Error: {e}")
                        if "exceeded" in str(e):
                            print("Sleeping for 3 seconds")
                            succuss = True
                            time.sleep(3)
                        else:
                            succuss = True
                            response = {"error_message": str(e)}
                            print(response)
                
                try:
                    answer = gpt_query.process_ai_message(response)
                except Exception as e:
                    answer = str(e)
                    print(answer)
                    
                gu.save_response(sub_save_path, answer)

                
                while True:
                    data = gu.get_finished_task(log_path)
                    if gpt_task_name in data.keys():
                        return 0               
                    if os.path.exists(os.path.join(sub_save_path, 'feedback.json')):
                        break
                    time.sleep(1)
                    
                while True:
                    feedbackinfo = os.stat(os.path.join(sub_save_path, 'feedback.json')) 
                    if feedbackinfo.st_size > 0:
                        break
                    
                data = gu.get_finished_task(log_path)
                if gpt_task_name in data.keys():
                    return 0

                with open(os.path.join(sub_save_path, 'feedback.json')) as f:
                    data = json.load(f) 
                
                main_task_flag = data['main_succeed']      
                if data['critic'] == 'succeed':
                    print('Task succeed!')
                    gpt_query.record_history(subtask=data['subtask'], code=data['code'], error=data['error'])
                    break
                else:
                    if data['reset']:
                        gpt_query.record_history(subtask=answer['subtask'], code=answer['code'], error=data['error'])
                        break
                    else:
                        gpt_query.record_history(subtask=answer['subtask'], code=answer['code'], error=data['error'])
                        break
            
            # reset parameters
            subtask_iter += 1
            
            #write json
            if subtask_iter > sub_num - 1:
                print(f"already attempt {subtask_iter} time, it is too long!")
                return 0

            if main_task_flag:    
                return 0


if __name__ == "__main__":
    args = parse_args()
    gpt_process(args)