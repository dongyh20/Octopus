import cv2
import json
import math
import random
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import omnigibson as og
from omnigibson.utils.vision_utils import segmentation_to_rgb
from omnigibson import object_states
import omnigibson.utils.transform_utils as T
from omnigibson.object_states.factory import (
    get_default_states,
    get_state_name,
    get_states_for_ability,
    get_states_by_dependency_order,
    get_texture_change_states,
    get_fire_states,
    get_steam_states,
    get_visual_states,
    get_texture_change_priority,
)
from bddl.object_taxonomy import ObjectTaxonomy


OBJECT_TAXONOMY = ObjectTaxonomy()


def get_robot_pos(obj):
    obj_pos, obj_ori = obj.get_position_orientation()
    vec_standard = np.array([0, -1, 0])
    rotated_vec = Quaternion(obj_ori[[3, 0, 1, 2]]).rotate(vec_standard)
    bbox = obj.native_bbox
    robot_pos = np.zeros(3)
    robot_pos[0] = obj_pos[0] + rotated_vec[0] * bbox[1] * 0.5 + rotated_vec[0]
    robot_pos[1] = obj_pos[1] + rotated_vec[1] * bbox[1] * 0.5 + rotated_vec[1]
    robot_pos[2] = 0.25
    
    return robot_pos

def cal_dis(pos1, pos2):
    #calculate the distance between the two position
    return np.linalg.norm(pos1 - pos2)

def update_obj(env, robot, obj_name):
    # update objects position according to robot position
    robot_pos = robot.get_position()
    robot_pos[2] += robot.aabb_center[2]
    robot_pos[2] -=0.2 
    obj = env.scene.object_registry("name",obj_name)  
    obj.set_position(robot_pos)

def quaternion_multiply(q1, q2):
    # calculate the multiply of two quaternion
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])

def update_camera_pos(camera,robot,obj): # for ego view pictures
    pos = robot.get_position()
    cam_pos = get_camera_position(pos)
    camera.set_position(cam_pos)
    obj_pos = obj.get_position()
    cam_pos = get_camera_position(robot.get_position())
    direction = obj_pos - cam_pos
    direction /= np.linalg.norm(direction)

    # calculate the quaternion to get the correct camera position
    cam_forward = np.array([0, 0, -1])
    dir1 = np.array([0, 1, 0])
    rotation_axis = np.cross(cam_forward, dir1)
    rotation_angle = np.arccos(np.dot(cam_forward, dir1))
    q_ro1 = Quaternion(axis=rotation_axis, angle=rotation_angle)

    dir2 = np.append(direction[[0, 1]], 0)
    rotation_axis = np.cross(dir1, dir2)
    if np.isclose(np.linalg.norm(rotation_axis), 0):
        rotation_axis = (
            np.array([1, 0, 0])
            if np.allclose(dir1, np.array([0, 0, 1]))
            else np.cross(dir1, np.array([0, 0, 1]))
        )

    rotation_angle = np.arccos(np.dot(dir1, dir2))
    q_ro2 = Quaternion(axis=rotation_axis, angle=rotation_angle) * q_ro1

    rotation_axis = np.cross(dir2, direction)
    if np.isclose(np.linalg.norm(rotation_axis), 0):
        rotation_axis = (
            np.array([1, 0, 0])
            if np.allclose(dir1, np.array([0, 0, 1]))
            else np.cross(dir1, np.array([0, 0, 1]))
        )

    rotation_angle = np.arccos(np.dot(dir2, direction))
    q_rotation = Quaternion(axis=rotation_axis, angle=rotation_angle) * q_ro2

    new_cam_ori = q_rotation.elements[[1, 2, 3, 0]]
    camera.set_orientation(new_cam_ori)

def get_camera_position(p):
    p[2] += 1.2
    # p[0] += 0.2
    return p

reverse = lambda states:{value:key for key,value in states.items()}
unary_states = {
    object_states.Cooked: "cookable",
    object_states.Burnt: "burnable",
    object_states.Frozen: "freezable",
    object_states.Heated: "heatable",
    object_states.Open: "openable",
    object_states.ToggledOn: "toggleable",
    object_states.Folded: "foldable",
    object_states.Unfolded: "unfoldable"
    }
binary__states={
    object_states.Inside: "inside",
    object_states.NextTo: "nextto",
    object_states.OnTop: "ontop",
    object_states.Under: "under",
    object_states.Touching: "touching",
    object_states.Covered: "covered",
    object_states.Contains: "contains",
    object_states.Saturated: "saturated",
    object_states.Filled: "filled",
    object_states.AttachedTo: "attached",
    object_states.Overlaid: "overlaid",
    object_states.Draped: "draped"
}

reversed_unary_states, reversed_binary_states = reverse(unary_states), reverse(binary__states)

def change_states(obj, states, oper):
    '''
    obj (Objects): The object that the states are needed to be changed.
    states (str): The specific states to be changed.
    oper (int): 0 or 1, meaning the False or True of the states.
    '''
    try:
        states_status=reversed_unary_states[states]
        obj.states[states_status].set_value(oper)
    except:
        raise Exception(f'Wrong state or operation {states, oper}')
        
def get_states(env,obj:str,state:str)->object_states:
    whole_dict={**reversed_unary_states,**reversed_binary__states}
    class_obj=env.scene.object_registry("name", obj)
    try:
        if whole_dict[state] in list(class_obj.states.keys()):
            return whole_dict[state]
        else:
            print(f"{obj} don't have states {whole_dict[state]}")
            raise Exception
    except:
        print(f"Wrong state {state}")
        raise Exception
