import cv2
import math
import json
import random
import numpy as np
from collections import OrderedDict

import omnigibson
from omnigibson.utils.vision_utils import segmentation_to_rgb
from omnigibson import object_states
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from bddl.object_taxonomy import ObjectTaxonomy
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

import utils.action_utils as au


OBJECT_TAXONOMY = ObjectTaxonomy()

def EasyGrasp(robot, obj, epsilon):
    #Grasp the robot within epsilon
    robot_pos = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(robot_pos, obj_pose)
    if dis < epsilon:
        robot_pos[2] += robot.aabb_center[2]
        robot_pos[2] -=0.2
        obj.set_position(robot_pos)
        robot.inventory.append(obj._name)
        print(f"now we have:{robot.inventory}")
    else:
        raise Exception(f"Cannot Grasp! robot is not within {epsilon} meters of {obj}")

def MoveBot(env, robot,obj,camera):
    pos = au.get_robot_pos(obj)
    robot.set_position(pos)
    au.Update_camera_pos(camera,robot,obj)
    if robot.inventory:
        # relationship between name and variable.
        for obj in robot.inventory: 
            au.update_obj(env, robot, obj)
            
def donothing(env):
    action = np.zeros(11)
    dumbact = OrderedDict([('robot0', action)])
    step = 0
    for _ in range(30):
        # og.sim.step()
        env.step(dumbact)
        step += 1
    
def registry(env, obj_name):
    return env.scene.object_registry("name", obj_name)

def cook(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot cook! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'cookable', 1)

def burn(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot burn! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'burnable', 1)

def freeze(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot freeze! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'freezable', 1)

def heat(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot heat! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'heatable', 1)

def open(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot open! robot is not within {epsilon} meters of {obj}")
    if obj.states[omnigibson.object_states.open.Open].get_value()==True:
        raise Exception(f"the {obj._name} has been opened! Please pay attention to the states of Observed Objects!!")
    au.change_states(obj, 'openable', 1)

def close(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot close! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'openable', 0)

def fold(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot fold! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'foldable', 1)

def unfold(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot unfold! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'unfoldable', 1)

def toggle_on(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot toggle on! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'togglable', 1)

def toggle_off(robot, obj, epsilon):
    bot_pose = robot.get_position()
    obj_pose = obj.get_position()
    dis = au.cal_dis(bot_pose, obj_pose)
    if dis > epsilon:
        raise Exception(f"Cannot toggle off! robot is not within {epsilon} meters of {obj}")
    au.change_states(obj, 'togglable', 0)
    
def put_inside(robot, obj1, obj2, epsilon):
    """
    put obj1 inside obj2
    """
    obj2_pos = obj2.get_position()
    dis = au.cal_dis(obj2_pos, robot.get_position())
    if dis < epsilon:
        obj1.set_position(obj2.get_position())
        robot.inventory.remove(obj1._name)
        print(f"the robot put {obj1._name} inside {obj2._name},now we have:{robot.inventory}")
    else:
        raise Exception(f"Cannot Put Inside! robot is not within {epsilon} meters of {obj2}")
    
def put_ontop(robot, obj1, obj2, epsilon):
    """
    put obj1 ontop obj2
    """
    obj2_pos = obj2.get_position()
    dis = au.cal_dis(obj2_pos, robot.get_position())
    if dis < epsilon:
        p_pos = obj2.get_position()
        p_pos[2] += 0.5 * obj1.native_bbox[2] + 0.5 * obj2.native_bbox[2]
        obj1.set_position(p_pos)
        robot.inventory.remove(obj1._name)
        print(f"the robot put {obj1._name} ontop {obj2._name},now we have:{robot.inventory}")
    else:
        raise Exception(f"Cannot Put Ontop! robot is not within {epsilon} meters of {obj2}")
    