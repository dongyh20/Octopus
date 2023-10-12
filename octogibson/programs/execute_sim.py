import os
import json
import yaml
import time
import sys
import argparse
import importlib
from importlib import reload

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options

from utils.robot_utils import *
import utils.sim_utils as su
from initial_pipeline import *

def parse_args():
    description = "EVLM_sim_process"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('idx', type=int, help='Index for data entry')
    parser.add_argument('process_path', type=str, default="/PATH/TO/CONFIG", help='path for loading process config')
    parser.add_argument("-l", "--/log/level", type=str, help="error log 1", required=True)
    parser.add_argument("-f", "--/log/fileLogLevel", type=str, help="error log 2", required=True)
    parser.add_argument("-o", "--/log/outputStreamLevel", type=str, help="error log 3", required=True)
    return parser.parse_args()


def sim_process(args):
    idx = args.idx
    process_config = args.process_path
    process_cfg = yaml.load(open(process_config, "r"), Loader=yaml.FullLoader)
    sub_num = process_cfg["process"]["sub_num"]
    log_path = process_cfg["process"]["log_path"]
    data_path = process_cfg["process"]["data_path"]
    task_path = process_cfg["process"]["task_path"]
    config_path = process_cfg["process"]["config_path"]
    
    try:
        with open(task_path) as f:
            data = json.load(f)
            
        EVLM_name = sorted(list(data))[idx]
        if "train" in data[EVLM_name]['split']:
            # init            
            task_data = data[EVLM_name]
            task_name = task_data['task_name']
            gpt_task_name = task_data['detailed_name']
            scene_name = task_data['env']
            removed_items = task_data['removed_item']
            main_target_states = task_data['target_states']
            og.log.info(f"EVLM:{EVLM_name}")
            og.log.info(f"scene:{scene_name}") # scene name to load the correponding bddl task
            og.log.info(f"task_name:{task_name}") # task name to load the correponding bddl task
            og.log.info(f"detailed_name:{gpt_task_name}") # detailed task name for data collection with GPT-4
            og.log.info(f'targets:{main_target_states}')
            save_path = su.f_mkdir(os.path.join(data_path, gpt_task_name))
            
            code_headings = su.get_headings()
            
            # create configs
            cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
            cfg["task"]["online_object_sampling"] = False
            cfg["scene"]["scene_model"] = scene_name
            cfg["task"]["activity_name"] = task_name
            og.log.info(cfg)
            
            # Load the environment
            try:
                env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)
            except:
                og.log.info("Loading Environment Error!")
                su.update_log(log_path, gpt_task_name, "error")
                return 0
            
            # set robot and camera
            robot = env.robots[0]
            camera= og.sim.viewer_camera 
            bbox_modalities = ["bbox_2d_loose"]
            for bbox_modality in bbox_modalities:
                camera.add_modality(bbox_modality)
            camera.focal_length = 10.
            
            # main task loop
            subtask_iter = 1
            while True:     
                main_succeed = False
                
                while True:
                    if os.path.exists(os.path.join(save_path, f"subtask_{subtask_iter}")):
                        break
                sub_save_path = os.path.join(save_path, f"subtask_{subtask_iter}")
                init_pipeline(env, camera, task_name=str(gpt_task_name), file_name=sub_save_path, removed_items=removed_items)
                response_path = os.path.join(sub_save_path, 'response.json')
                feedback_path = os.path.join(sub_save_path, 'feedback.json')
                
                #subtask loop
                while True:  
                    reset = False
                    error=''
                    # wait for the GPT response
                    while True:
                        response_info = os.stat(response_path) 
                        if response_info.st_size > 0:
                            break
                    with open(response_path) as f:
                        data = json.load(f)
                    answer = data['response']
                    # get error response
                    if isinstance(answer, str):
                        subtask = ""
                        code = ""
                        error = answer
                        critic = 'fail'
                        reset = False
                        su.update_log(log_path, gpt_task_name, "failed")
                        su.save_feedback(feedback_path, subtask, code, error, critic, reset, main_succeed)
                        env.close()
                        return 0
                    else:
                        subtask, code = answer['subtask'], answer['code']
                        with open(os.path.join(data_path, f"{gpt_task_name}/subtask_{subtask_iter}/action.py"), 'w') as f:
                            f.write(code_headings)
                            f.write(code)
                        
                        sys.path = list(set(sys.path))
                        if subtask_iter != 1:
                            sys.path.remove(os.path.join(data_path, f"{gpt_task_name}/subtask_{subtask_iter-1}"))
                        sys.path.append(os.path.join(data_path, f"{gpt_task_name}/subtask_{subtask_iter}"))
                        import action
                        time.sleep(1)
                        
                        try:
                            reload(action)
                            time.sleep(2)
                            action.act(robot,env,camera)
                            og.log.info("act...")
                        except Exception as e:
                            error = str(e)
                            env.reset()
                            
                            # reset parameters
                            robot.inventory = []
                            robot.visible_only=True
                            subtask = subtask
                            code = code
                            error = error
                            critic = 'fail'
                            reset = True
                            break
                    
                    # subtask verification
                    target_states = answer
                    for obj in target_states['obj_2']:
                        value = su.verify_obj_2(env,obj[0], obj[1], obj[2])
                        if not value:
                            error += f"State {obj[1]} of object {obj[0]} is not {obj[2]}\n"
                    #verify binary states
                    for obj in target_states['obj_3']:
                        if obj[0] == "robot" or obj[2] == "robot":
                            continue
                        value = su.verify_obj_3(env,obj[0], obj[1], obj[2],obj[3], os.path.join(data_path, f"{gpt_task_name}/subtask_{subtask_iter}/action.py"))
                        if not value:
                            error += f"{obj[0]} is not {obj[1]} {obj[2]}\n"

                    if len(error) == 0:
                        subtask = subtask
                        error = error
                        critic = 'succeed'
                        reset = False
                        break
                    else:
                        subtask = subtask
                        error = error
                        critic = 'fail'
                        reset = True
                        env.reset()
                        robot.inventory=[]
                        break
                    
                subtask_iter += 1

                # verify the whole task
                if critic == 'succeed':
                    signal=[]
                    target=main_target_states

                    for tar in target:
                        if len(tar) == 3:
                            if su.verify_taskgoal(env,*tar):
                                signal.append(1)
                            else:
                                signal.append(0)
                        elif len(tar) == 4:
                            if su.verify_binary_taskgoal(env,*tar):
                                signal.append(1)
                            else:
                                signal.append(0)
                    if 0 not in signal:
                        main_succeed = True
                        su.update_log(log_path, gpt_task_name, "succeed")                     
                        og.log.info(f"finish {task_name}! congrats!")
                        su.save_feedback(feedback_path, subtask, code, error, critic, reset, main_succeed)
                        env.close()
                        return 0
                    else:
                        su.save_feedback(feedback_path, subtask, code, error, critic, reset, main_succeed)
                else:
                    su.save_feedback(feedback_path, subtask, code, error, critic, reset, main_succeed)
                
                # retrieve for previous actions
                if reset:
                    if subtask_iter != 1:
                        for iter_num in range(1,subtask_iter):
                            path = os.path.join(data_path, f"{gpt_task_name}/subtask_{iter_num}")
                            with open(os.path.join(path,"feedback.json"))as f:
                                tmp_feedback = json.load(f)
                            if tmp_feedback['critic'] == 'succeed':
                                time.sleep(1)
                                data_path_ = data_path.replace('/', '.')
                                module=importlib.import_module(f"{data_path_}.{gpt_task_name}.subtask_{iter_num}.action")
                                og.log.info(f"{gpt_task_name}.subtask_{iter_num}.action retrieve")
                                module.act(robot,env,camera)   
                
                if subtask_iter > sub_num - 1:
                    og.log.info(f"already attempt {subtask_iter} time, it is too long!")
                    su.update_log(log_path, gpt_task_name, "failed")   
                    env.close()        
                    return 0
    except:
        og.log.info(f"loop failed")
        su.update_log(log_path, gpt_task_name, "error")   
        return 0

if __name__ == "__main__":
    args = parse_args()
    sim_process(args)