import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.sensors import VisionSensor
from omnigibson import object_states
import action_list as al
import utils.robot_utils as ru

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_OBJECT_STATES = True

def init_pipeline(env, camera, task_name, file_name=None, removed_items=None):
    iter = 0
    cap = ru.Capture_System(robot=env.robots[0], camera=camera, env=env, filename=file_name, TASK=task_name, removed_items=removed_items)
    robot = env.robots[0]

    robot.visible=False
    robot.visible_only=True
    
    # set camera position
    action = np.zeros(11)
    ppposition = robot.get_position()
    cam_position = ru.get_camera_position(ppposition)
    robot_sensor = robot._sensors['robot0:eyes_Camera_sensor']
    rs_p, rs_o = robot_sensor.get_position_orientation()
    origin_pos_ori = [cam_position,rs_o].copy()
    cap.setposition(cam_position, rs_o)

    al.donothing(env, action)
    print("ego:detect surroundings!")
    
    # EGOs
    for _ in range(8):
        cap.FlyingCapture(f'{iter}_detect_surroundings')   
        iter+=1   
        rs_o = ru.trans_camera(rs_o)
        cap.setposition(cam_position, rs_o)
        al.donothing(env, action) 
        
    cap.collectdata_v2(robot)
    robot.visible=False
    robot.visible_only=True
    al.donothing(env, action)
    
    # BEVs
    ppposition = robot.get_position()
    cam_position = ru.get_camera_position_bev(ppposition)
    cap.setposition(cam_position, ru.trans_camera(robot.get_orientation()))
    al.donothing(env, action)
    cap.FlyingCapture(f'{iter}_BEV_surroundings')   
    iter += 1  
    
    cam_position[2] += 2
    cap.setposition(cam_position, ru.trans_camera(robot.get_orientation()))
    al.donothing(env, action)
    cap.FlyingCapture(f'{iter}_BEV_surroundings')   
    iter+=1  
    al.donothing(env, action)
    cap.collectdata_v2(robot)

    cap.setposition(*origin_pos_ori)
    al.donothing(env, action)

    cap.writejson()