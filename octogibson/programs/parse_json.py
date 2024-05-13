import json


def all_SG_parse_json(path, object_file):
    SG = []
    OBJ = []
    OBJ_INFO = []
    OBS = []
    with open(path) as f:
        data = json.load(f)
    for k in data.keys():
        parsed_data = data[k]
        if type(parsed_data)==str:
            TASK = parsed_data
            continue
        for ks in parsed_data.keys(): #each object in data
            if ks == 'scene_graph':
                all_relations=[]
                with open(path[:-10] + "psg_relation.json","r")as f:
                    all_psg=json.load(f)
                for key in all_psg.keys():
                    for relation in all_psg[key]:
                        if relation not in all_relations:
                            all_relations.append(relation)
                
                for s in all_relations:
                    if s not in SG:
                        SG.append(tuple(s))
                        
            elif ks == 'inventory':
                INV = parsed_data[ks]
            else:
                if ks not in OBJ:
                    obj = []
                    obj.append(ks)
                    obj_data = parsed_data[ks]
                    obj.append(tuple(obj_data['ability']))
                    obj.append(tuple(obj_data['position_in_bot'])) # correct the key
                    OBJ_INFO.append(obj)
                    OBJ.append(ks)
                    obs = (ks, tuple(obj_data['ability']),obj_data['position_in_bot']['rho'])
                    OBS.append(obs)
                else:
                    continue    
    sg_str = ''
    for i in SG:
        s = ','.join(i)
        sg_str += '(' + s + ')'
    obs_str = ''
    for i in OBS:
        s = '(' + i[0] + ', ' + str(i[1])+ ", " +str(round(i[2], 2)) + ')'
        obs_str += s 

    EVLM_obj = path.split("/")[-3]
    with open(object_file,"r")as f: # remove the redundant data according to the object_file 
        all_obj = json.load(f)
    for ele in all_obj[EVLM_obj]:
        obs_str += f"({ele}),"
    final_SG=list(set(SG)).copy()
    for (obj1, prep, obj2) in list(set(SG)):
        if (obj2, prep, obj1) in final_SG:
            final_SG.remove((obj1,prep,obj2))

    return [final_SG, obs_str, str(INV), str(TASK)]

def parse_json(path):
    SG = []
    OBJ = []
    OBJ_INFO = []
    OBS = []
    with open(path) as f:
        data = json.load(f)
    for k in data.keys():
        parsed_data = data[k]
        if type(parsed_data)==str:
            TASK=parsed_data
            continue
        for ks in parsed_data.keys(): #each object in data
            if ks == 'scene_graph':
                sg = parsed_data[ks]
                for s in sg:
                    if s not in SG:
                        SG.append(tuple(s))
            elif ks == 'inventory':
                INV = parsed_data[ks]
            else:
                if ks not in OBJ:
                    obj = []
                    obj.append(ks)
                    obj_data = parsed_data[ks]
                    obj.append(tuple(obj_data['ability']))
                    obj.append(tuple(obj_data['position_in_bot'])) 
                    OBJ_INFO.append(obj)
                    OBJ.append(ks)
                    obs = (ks, tuple(obj_data['ability']),obj_data['position_in_bot']['rho'])
                    OBS.append(obs)
                else:
                    continue    
    sg_str = ''
    for i in SG:
        s = ','.join(i)
        sg_str += '(' + s + ')'
    obs_str = ''
    for i in OBS:
        s = '(' + i[0] + ', ' + str(i[1])+ ", " +str(round(i[2], 2)) + ')'
        obs_str += s

    final_SG = list(set(SG)).copy()
    for (obj1,prep,obj2) in list(set(SG)): # remove the redundant data
        if (obj2,prep,obj1) in final_SG:
            final_SG.remove((obj1,prep,obj2))
            
    return [final_SG, obs_str, str(INV), str(TASK)]