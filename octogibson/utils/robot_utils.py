import os
import cv2
import math
import json
import random
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from bddl.object_taxonomy import ObjectTaxonomy
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


OBJECT_TAXONOMY = ObjectTaxonomy()


def cal_dis(pos1, pos2):
    #calculate the distance between the two position
    return np.linalg.norm(pos1 - pos2)

def quaternion_multiply(q1, q2):
    # calculate the multiply of two quaternion
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])

def get_camera_position(p):
    p[2] += 1.2
    # p[0] += 0.2
    return p

def get_camera_position_bev(p):
    p[2] += 3
    return p

def trans_camera(q):
    random_yaw = np.pi / 4
    yaw_orn = R.from_euler("Z", random_yaw)
    new_camera_orn = quaternion_multiply(yaw_orn.as_quat(), q)
    # print(new_camera_orn)
    return new_camera_orn


# class of flying camera
class Capture_System():
    def __init__(self, robot, camera, env, filename,TASK, position=np.array([-2.48302418,  1.55655398,  2.22882511]),orientation=np.array([ 0.56621324, -0.0712958 , -0.10258276,  0.81473692]), removed_items=None):
        self.robot = robot
        self.camera = camera
        self.env = env
        self.camera.set_position_orientation(position, orientation)
        self.seglist = []
        self.instancemap = []
        self.FILENAME = filename
        self.removed_items = removed_items
        self.allobject = self._getallobject()
        self.result_json = {}
        self.actionlist = [] #check the action to appear only once
        self.OG_results = self._decomposed()
        self.blacklist = ["walls","electric_switch","floors","ceilings","window"]
        self.task = TASK
    
    def _getallobject(self):
        try:
            objectlist=[i['name'] for i in self.env.objects_config]
        except:
            pass
        scenedict=self.env.scene.get_objects_info()
        scenelist=list(scenedict['init_info'].keys())
        return list(set(objectlist+scenelist))
 
    def setposition(self,position=None,orientation=None):
        if type(orientation)==np.ndarray and type(position)!=np.ndarray:
            self.camera.set_orientation(orientation)
        if type(orientation)!=np.ndarray and type(position)==np.ndarray:
            self.camera.set_position(position)
        if  type(orientation)==np.ndarray and type(position)==np.ndarray:
            self.camera.set_position_orientation(position=position,orientation=orientation)
        else:
            raise TypeError
        
    def xyz2spherical(self,position):
        x, y, z = [position[i] for i in range(3)]
        rho = math.sqrt(x**2 + y**2 + z**2)
        theta = math.atan2(y, x)
        phi = math.acos(z/rho)

        #convert to degree
        theta_deg = math.degrees(theta)
        phi_deg = math.degrees(phi)
        return {"rho":rho,"theta_deg":theta_deg,"phi_deg":phi_deg}

    def set_in_rob(self,robot):
        obj_in_rob = {}
        for object_name in self.allobject:
            tempobj = self.env.scene.object_registry("name",object_name)
            pose_in_rob = self.align_coord(tempobj,robot)
            obj_in_rob[object_name] = (self.xyz2spherical(pose_in_rob[0]),pose_in_rob[1])
        return obj_in_rob
                
    def align_coord(self,obj,rob):
        # convert world2bot coord
        obj_in_world = T.pose2mat(obj.get_position_orientation())
        rob_in_world = T.pose2mat(rob.get_position_orientation())
        world_in_rob = T.pose_inv(rob_in_world)
        obj_in_rob = T.pose_in_A_to_pose_in_B(obj_in_world, world_in_rob)
        obj2rob = T.mat2pose(obj_in_rob)
        return obj2rob

    def FlyingCapture(self,iter,file_name=None):
        obs_dict = self.camera._get_obs()
        for modality in ["rgb", "depth", "seg_instance"]:
            query_name = modality
            if query_name in obs_dict:
                if modality == "rgb":
                    pass
                elif modality == "seg_instance":
                    # Map IDs to rgb
                    self.instancemap.append({f"{self.FILENAME}/"+query_name + f'{iter}.png':obs_dict[query_name][0]})
                    segimg = segmentation_to_rgb(obs_dict[query_name][0], N=256)
                    instancemap = obs_dict[query_name][1]
                    for item in instancemap:
                        bbox_2ds = obs_dict["bbox_2d_loose"] #
                        hextuple = [f"{self.FILENAME}/"+query_name + f'{iter}.png',item[1].split("/")[-1],item[3],item[0],'','']
                        for bbox_2d in bbox_2ds:
                            if bbox_2d[0]==item[0]:
                                bbox2d_info = [bbox_2d[i] for i in range(6,10,1)]
                                hextuple[4] = bbox2d_info
                                break
                        self.seglist.append(hextuple)
                elif modality == "normal":
                    # Re-map to 0 - 1 range
                    pass
                else:
                    # Depth, nothing to do here
                    pass
                if modality == "seg_instance":
                    rgbimg = cv2.cvtColor(segimg, cv2.COLOR_BGR2RGB)
                elif modality == "rgb":
                    rgbimg = cv2.cvtColor(obs_dict[query_name], cv2.COLOR_BGR2RGB)
                else:
                    rgbimg = obs_dict[query_name]

                if file_name is not None:
                    cv2.imwrite(query_name + str(file_name) + '.png', rgbimg)
                else:
                    path=os.path.dirname(f"{self.FILENAME}/"+query_name + f'{iter}.png')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    
                    cv2.imwrite(f"{self.FILENAME}/"+query_name + f'{iter}.png', rgbimg)
                    print(f"save as: {self.FILENAME}/"+query_name + f'{iter}.png')
        
    def parsing_segmentdata(self): #parse all data from the files that we have collected
        seglists = self.seglist
        instancemaps = self.instancemap
        parse = lambda path:list(path.keys())[0]
        self.nowwehave = []
        for ele in instancemaps:
            path = parse(ele)
            templist = []
            tempdict = {}
            map = ele[path]
            instance = list(np.unique(map))

            for seglist in seglists:
                if seglist[0]==path:
                    instance_id = seglist[3]
                    if instance_id in instance:
                        templist.append(seglist[1])
            tempdict[path] = templist
            self.nowwehave.append(tempdict) #dict{path:object_name}
        print("now begin to parse segmentation data")
        return self.nowwehave
    
    def parseSG(self,objects):
        pairs=[]
        SG=[]
        for i in range(len(objects)):
            for j in range(len(objects)):
                cnt=0
                if(objects[i]!=objects[j]):
                    for blackele in self.blacklist:
                        if blackele in objects[i]:
                            cnt+=1
                        if blackele in objects[j]:
                            cnt+=1
                    if cnt!=2:
                        pairs.append((objects[i],objects[j]))
        reduced_pairs=[]
        for pair in pairs: 
            tempscore=0 #record whether the object in og_results
            for ele_pair in pair:
                if ele_pair in self.OG_results:
                    tempscore+=1
            if tempscore!=0:
                reduced_pairs.append(pair)


        for pair in reduced_pairs:
            obj0=self.env.scene.object_registry("name",pair[0])
            obj1=self.env.scene.object_registry("name",pair[1])
            try:
                is_inside=obj0.states[object_states.Inside].get_value(obj1)
                if is_inside:
                    SG.append((obj0._name,"inside",obj1._name))
            except:
                pass         
            try:
                is_ontop=obj0.states[object_states.OnTop].get_value(obj1)
                if is_ontop:
                    SG.append((obj0._name,"ontop",obj1._name))                   
            except:
                pass        
            try:
                is_overlaid=obj0.states[object_states.Overlaid].get_value(obj1)
                if is_overlaid:
                    SG.append((obj0._name,"overlaid",obj1._name))                   
            except:
                pass        
            try:
                is_under=obj0.states[object_states.Under].get_value(obj1)
                if is_under:
                    SG.append((obj0._name,"under",obj1._name))                   
            except:
                pass
        temp_SG = SG.copy()
        for (obj1,prep,obj2) in SG:
            if (obj2,prep,obj1) in temp_SG:
                temp_SG.remove((obj1,prep,obj2))
        return {"scene_graph":SG}

    def _decomposed(self): #decomposed all the object in the env at the very beginning
        OG_results = []
        parsed_objects = self.env.task.activity_conditions.parsed_objects
        OG_dict = self.env.task.load_task_metadata()["inst_to_name"] # format in OG. floor.n.01_1 -> floors_hcqtge_0
        for key in parsed_objects.keys():
            for ele in parsed_objects[key]: # bacon.n.01_1
                if ele not in self.removed_items:
                    OG_results.append(OG_dict[ele])
        return OG_results

    def collectdata_v2(self,robot): #each time change the robot position need to collectdata
        self.result_json['task']=self.task
        nowwehave=self.parsing_segmentdata()
        inventory=self.robot.inventory.copy()
        sub_nowwehave=[]
        for key in nowwehave:
            if list(key.keys())[0].split("/")[-1].rstrip('.png').lstrip("seg_instance") not in self.actionlist:
                sub_nowwehave.append(key)
        seglists=self.seglist
        obj_in_robs=self.set_in_rob(robot) #the object in now robot_pos
        obj_metadata={} #get the object metadata
        robot_pose=robot.get_position().copy()
        editable_states={object_states.Cooked:"cookable",object_states.Burnt:"burnable",object_states.Frozen:"freezable",object_states.Heated:"heatable",
                         object_states.Open:"openable",object_states.ToggledOn:"togglable",object_states.Folded:"foldable",object_states.Unfolded:"unfoldable"}
        
        blacklist=['robot0']
        for ele in self.OG_results:
            for a in self.blacklist:
                if a in ele:
                    blacklist.append(ele)

        for ele in sub_nowwehave:
            picpath=list(ele.keys())[0]
            objects=list(ele.values())[0]
            intersect_objects=list(set(self.OG_results)-set(blacklist)) #TODO
            action=picpath.split("/")[-1][12:-4]
            scene_graph=self.parseSG(intersect_objects) #TODO
            if action not in self.actionlist:
                self.actionlist.append(action)
            obj_metadata.clear()
            if len(intersect_objects)==0:
                self.result_json[action]={}
                continue
            for obj_name in intersect_objects:
                # ability=OBJECT_TAXONOMY.get_abilities(OBJECT_TAXONOMY.get_synset_from_category(obj_name.split("_")[0]))
                object=self.env.scene.object_registry("name",obj_name)
                if object== None:
                    continue
                obj_metadata[obj_name]={}
                states={"ability":[(editable_states[sta],int(object.states[sta]._get_value())) for sta in list(object.states.keys()) if sta in editable_states.keys()]}
                obj_metadata[obj_name].update(states.copy())
                obj_in_rob=obj_in_robs[obj_name]
                position_in_bot={"position_in_bot":obj_in_rob[0]}
                self.result_json[action]={}
                obj_metadata[obj_name].update(position_in_bot)
                orientation={"orientation_in_bot":obj_in_rob[1].tolist()}

                obj_metadata[obj_name].update(orientation)
                position_in_world={"position_in_world":object.get_position().tolist()}
                obj_metadata[obj_name].update(position_in_world)

                bot_pose={"bot_in_world":robot_pose.tolist()}
                obj_metadata[obj_name].update(bot_pose)
                path={"path":picpath}
                obj_metadata[obj_name].update(path)

                self.result_json[action].update(obj_metadata)
                self.result_json[action].update(scene_graph)

                inventory_dict={"inventory":inventory} #TODO check this choiszt
                self.result_json[action].update(inventory_dict)
        return self.result_json

    def writejson(self):
        with open(f"{self.FILENAME}/task1.json","w")as f:
            f.write(json.dumps(self.result_json, indent=4))