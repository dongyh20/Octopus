import re
import time

import utils as U
# from javascript import require
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from prompts import load_prompt
from control_primitives_context import load_control_primitives_context
import utils.gpt_utils as gu 

import os
import openai
# openai.api_type = "azure"
# openai.api_base = "https://voyager.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
# openai.api_key = "xxxx"

# def gpt_request(content):
    
#     print(response['choices'][0]['message']['content'])
#     return response['choices'][0]['message']['content']
class OctopusAgent:
    def __init__(
        self,
        model_name="gpt-4-0125-preview",
        temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            print(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
        else:
            self.chest_memory = {}
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )
        self.history_info={}
        self.record_history()

    def gpt_request(self,content):
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            engine="voyager",
            messages = [{"role":"user","content":content}],
            temperature=0.6,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
            )
        # print(response['choices'][0]['message']['content'])
        # return response['choices'][0]['message']['content']
        return response['choices'][0]['message']['content']

    def render_system_message(self):
        system_template = gu.load_prompt("prompt_template")

        response_format = gu.load_prompt("response_template")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(response_format=response_format)
        assert isinstance(system_message, SystemMessage)
        return system_message
    
    def record_history(self, subtask="", code="", error=""):
        self.history_info['subtask'] = subtask
        self.history_info['code'] = code
        self.history_info['error'] = error    
    
        # scene_graph, object, inventory, task = human_info

    # def render_human_message(
    #     self, events, code="", task="", context="", critique=""
    # ):
    #     chat_messages = []
    #     error_messages = []
    #     # FIXME: damage_messages is not used
    #     damage_messages = []
    #     assert events[-1][0] == "observe", "Last event must be observe"
    #     for i, (event_type, event) in enumerate(events):
    #         if event_type == "onChat":
    #             chat_messages.append(event["onChat"])
    #         elif event_type == "onError":
    #             error_messages.append(event["onError"])
    #         elif event_type == "onDamage":
    #             damage_messages.append(event["onDamage"])
    #         elif event_type == "observe":
    #             biome = event["status"]["biome"]
    #             time_of_day = event["status"]["timeOfDay"]
    #             voxels = event["voxels"]
    #             entities = event["status"]["entities"]
    #             health = event["status"]["health"]
    #             hunger = event["status"]["food"]
    #             position = event["status"]["position"]
    #             equipment = event["status"]["equipment"]
    #             inventory_used = event["status"]["inventoryUsed"]
    #             inventory = event["inventory"]
    #             assert i == len(events) - 1, "observe must be the last event"
    #     observation = ""
    #     if code:
    #         observation += f"Code from the last round:\n{code}\n\n"
    #     else:
    #         observation += f"Code from the last round: No code in the first round\n\n"
    #     if self.execution_error:
    #         if error_messages:
    #             error = "\n".join(error_messages)
    #             observation += f"Execution error:\n{error}\n\n"
    #         else:
    #             observation += f"Execution error: No error\n\n"
    #     if self.chat_log:
    #         if chat_messages:
    #             chat_log = "\n".join(chat_messages)
    #             observation += f"Chat log: {chat_log}\n\n"
    #         else:
    #             observation += f"Chat log: None\n\n"
    #     observation += f"Biome: {biome}\n\n"
    #     observation += f"Time: {time_of_day}\n\n"
    #     if voxels:
    #         observation += f"Nearby blocks: {', '.join(voxels)}\n\n"
    #     else:
    #         observation += f"Nearby blocks: None\n\n"

    #     if entities:
    #         nearby_entities = [
    #             k for k, v in sorted(entities.items(), key=lambda x: x[1])
    #         ]
    #         observation += f"Nearby entities (nearest to farthest): {', '.join(nearby_entities)}\n\n"
    #     else:
    #         observation += f"Nearby entities (nearest to farthest): None\n\n"

    #     observation += f"Health: {health:.1f}/20\n\n"

    #     observation += f"Hunger: {hunger:.1f}/20\n\n"

    #     observation += f"Position: x={position['x']:.1f}, y={position['y']:.1f}, z={position['z']:.1f}\n\n"

    #     observation += f"Equipment: {equipment}\n\n"

    #     if inventory:
    #         observation += f"Inventory ({inventory_used}/36): {inventory}\n\n"
    #     else:
    #         observation += f"Inventory ({inventory_used}/36): Empty\n\n"

    #     if not (
    #         task == "Place and deposit useless items into a chest"
    #         or task.startswith("Deposit useless items into the chest at")
    #     ):
    #         observation += self.render_chest_observation()

    #     observation += f"Task: {task}\n\n"

    #     if context:
    #         observation += f"Context: {context}\n\n"
    #     else:
    #         observation += f"Context: None\n\n"

    #     if critique:
    #         observation += f"Critique: {critique}\n\n"
    #     else:
    #         observation += f"Critique: None\n\n"

    #     return HumanMessage(content=observation)
            
    def parse_picinfo(self,info):
        import re
        result=''
        pattern = r"pic(\d+)"
        result += f'{re.search(pattern, info).group(0)}\n'
        pattern = r"yaw:\d+\.\d+"
        result+=f'direction={re.search(pattern, info).group(0)}\n'
        pattern = r"{(.*?)}"
        
        obj_with_dis = re.search(pattern, info).group(1)
        if len(obj_with_dis)==0:
            result='None'
            return result
        items = obj_with_dis.split(',')
        block_dict = {}
        for i in range(0, len(items), 2):
            block_type = items[i]
            value = items[i+1]
            if block_type not in block_dict:
                block_dict[block_type] = []
            block_dict[block_type].append(value)

        for block_type, values in block_dict.items():
            if len(values) > 2:
                values = sorted(values)[:2]
                block_dict[block_type]=values

        formatted_blocks = []
        for block_type, values in block_dict.items():
            formatted_values = ",".join(values)
            formatted_block = f"{block_type}({formatted_values})"
            formatted_blocks.append(formatted_block)

        result += " ".join(formatted_blocks)
        return result

    def render_human_message(self, current_data,task,critique):
        message = ""    
        error=""
        pic_info = []
        error_messages = []
        # FIXME: damage_messages is not used
        damage_messages = []
        assert current_data[-1][0] == "observe", "Last event must be observe"
        for i, (event_type, event) in enumerate(current_data):
            if event_type == "onChat":
                if event["onChat"].startswith("pic"): 
                    pic_info.append(event["onChat"])
            elif event_type == "onError":
                error_messages.append(event["onError"])
            elif event_type == "onDamage":
                damage_messages.append(event["onDamage"])
            elif event_type == "observe":
                biome = event["status"]["biome"]
                time_of_day = event["status"]["timeOfDay"]
                voxels = event["voxels"]
                entities = event["status"]["entities"]
                health = event["status"]["health"]
                hunger = event["status"]["food"]
                position = event["status"]["position"]
                equipment = event["status"]["equipment"]
                inventory_used = event["status"]["inventoryUsed"]
                inventory = event["inventory"]
                assert i == len(current_data) - 1, "observe must be the last event"



    #     observation += f"Equipment: {equipment}\n\n"


    #     if not (
    #         task == "Place and deposit useless items into a chest"
    #         or task.startswith("Deposit useless items into the chest at")
    #     ):
    #         observation += self.render_chest_observation()

    #     observation += f"Task: {task}\n\n"

    #     if context:
    #         observation += f"Context: {context}\n\n"
    #     else:
    #         observation += f"Context: None\n\n"

        if critique: #TODO the usage of critique
            message += f"Critique: {critique}\n\n"
        else:
            message += f"Critique: None\n\n"    

        
        message+=f"Observed Objects:\n"
        for info in pic_info:
            message+=f"{self.parse_picinfo(info)}\n"

        message += f"Task Goal: {task}\n"
        
        
        if len(self.history_info['subtask']) > 0:
            message += f"Original Subtasks: {self.history_info['subtask']}\n"
        else:
            message += f"Original Subtasks: None\n"

        if len(self.history_info['code']) > 0:
            message += f"Previous Action Code: {self.history_info['code']}\n"

            if self.history_info['error']: # planning error
                error = self.history_info['error']
            if error_messages: # simulator error
                error += "\n".join(error_messages)
            if len(self.history_info['error'])==0 and len(error_messages)==0:
                message += f"Execution Error: No error\n"  
            else:
                message += f"Execution Error:\n{error}\n\n"
            # if len(self.history_info['error']) > 0:
            #     message += f"Execution Error:{self.history_info['error']}\n"
            # else:
            #     message += f"Execution Error: No error\n"  
        elif len(self.history_info['code']) == 0: 
            message += f"Previous Action Code: No code\n"
            message += f"Execution error: No error\n"

        if inventory:
            message += f"Inventory: {inventory}\n"
        else:
            message += f"Inventory: Empty\n"
        # message += "Now, please output Explain, Subtasks (revise if necessary), Code that completing the next subtask, according to the instruction above. Remember you should give me just one subtask each turn and can only use the functions provided above and pay attention to the response format."
        message += "Now, please output Explain, Subtasks (revise if necessary), Code that completing the next subtask, according to the instruction above. Remember you should pay attention to the response format and give me just one subtask each turn ."    
        return HumanMessage(content=message)

    def update_chest_memory(self, chests):
        for position, chest in chests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    print(
                        f"\033[32mAction Agent removing chest {position}: {chest}\033[0m"
                    )
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    print(f"\033[32mAction Agent saving chest {position}: {chest}\033[0m")
                    self.chest_memory[position] = chest
        U.dump_json(self.chest_memory, f"{self.ckpt_dir}/action/chest_memory.json")

    def render_chest_observation(self):
        chests = []
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) > 0:
                chests.append(f"{chest_position}: {chest}")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) == 0:
                chests.append(f"{chest_position}: Empty")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, str):
                assert chest == "Unknown"
                chests.append(f"{chest_position}: Unknown items inside")
        assert len(chests) == len(self.chest_memory)
        if chests:
            chests = "\n".join(chests)
            return f"Chests:\n{chests}\n\n"
        else:
            return f"Chests: None\n\n"

    # def render_human_message(
    #     self, *, events, code="", task="", context="", critique=""
    # ):
    #     chat_messages = []
    #     error_messages = []
    #     # FIXME: damage_messages is not used
    #     damage_messages = []
    #     assert events[-1][0] == "observe", "Last event must be observe"
    #     for i, (event_type, event) in enumerate(events):
    #         if event_type == "onChat":
    #             chat_messages.append(event["onChat"])
    #         elif event_type == "onError":
    #             error_messages.append(event["onError"])
    #         elif event_type == "onDamage":
    #             damage_messages.append(event["onDamage"])
    #         elif event_type == "observe":
    #             biome = event["status"]["biome"]
    #             time_of_day = event["status"]["timeOfDay"]
    #             voxels = event["voxels"]
    #             entities = event["status"]["entities"]
    #             health = event["status"]["health"]
    #             hunger = event["status"]["food"]
    #             position = event["status"]["position"]
    #             equipment = event["status"]["equipment"]
    #             inventory_used = event["status"]["inventoryUsed"]
    #             inventory = event["inventory"]
    #             assert i == len(events) - 1, "observe must be the last event"

    #     observation = ""

    #     if code:
    #         observation += f"Code from the last round:\n{code}\n\n"
    #     else:
    #         observation += f"Code from the last round: No code in the first round\n\n"

    #     if self.execution_error:
    #         if error_messages:
    #             error = "\n".join(error_messages)
    #             observation += f"Execution error:\n{error}\n\n"
    #         else:
    #             observation += f"Execution error: No error\n\n"

    #     if self.chat_log:
    #         if chat_messages:
    #             chat_log = "\n".join(chat_messages)
    #             observation += f"Chat log: {chat_log}\n\n"
    #         else:
    #             observation += f"Chat log: None\n\n"

    #     observation += f"Biome: {biome}\n\n"

    #     observation += f"Time: {time_of_day}\n\n"

    #     if voxels:
    #         observation += f"Nearby blocks: {', '.join(voxels)}\n\n"
    #     else:
    #         observation += f"Nearby blocks: None\n\n"

    #     if entities:
    #         nearby_entities = [
    #             k for k, v in sorted(entities.items(), key=lambda x: x[1])
    #         ]
    #         observation += f"Nearby entities (nearest to farthest): {', '.join(nearby_entities)}\n\n"
    #     else:
    #         observation += f"Nearby entities (nearest to farthest): None\n\n"

    #     observation += f"Health: {health:.1f}/20\n\n"

    #     observation += f"Hunger: {hunger:.1f}/20\n\n"

    #     observation += f"Position: x={position['x']:.1f}, y={position['y']:.1f}, z={position['z']:.1f}\n\n"

    #     observation += f"Equipment: {equipment}\n\n"

    #     if inventory:
    #         observation += f"Inventory ({inventory_used}/36): {inventory}\n\n"
    #     else:
    #         observation += f"Inventory ({inventory_used}/36): Empty\n\n"

    #     if not (
    #         task == "Place and deposit useless items into a chest"
    #         or task.startswith("Deposit useless items into the chest at")
    #     ):
    #         observation += self.render_chest_observation()

    #     observation += f"Task: {task}\n\n"

    #     if context:
    #         observation += f"Context: {context}\n\n"
    #     else:
    #         observation += f"Context: None\n\n"

    #     if critique:
    #         observation += f"Critique: {critique}\n\n"
    #     else:
    #         observation += f"Critique: None\n\n"

    #     return HumanMessage(content=observation)

    def process_ai_message(self, processed_message):
        error = None
        # classes = ["Explain:", "Subtasks:", "Code:", "Target States:"]
        # classes = ["Explain:", "Subtasks:", "Code:", "Target States:"]
        classes = ["Explain:", "Subtasks:", "Code:"]
        idxs = []
        for c in classes:
            m = processed_message.find(c)
            idxs.append(m)
        if -1 in idxs:
            raise Exception('Invalid response format!')
    
        # parse process
        try:
            explain = processed_message[:idxs[1]]
            subtask = processed_message[idxs[1]:idxs[2]]
            code = processed_message[idxs[2]:]
            # code = processed_message[idxs[2]:idxs[3]]
            # target = processed_message[idxs[3]:]
            
            #EXPLAIN
            explain_str = explain.split('Explain:')[1]
            explain_str = explain_str.replace('\n', '')
            explain_str = explain_str.replace('\n\n', '')
            
            #SUBTASK
            subtask_str = subtask.split('Subtasks:')[1]
            subtask_str = subtask_str.replace('\n\n', '')
            
            #CODE
            code_str = code.split('```javascript\n')[1].split('```')[0]
            
            # #TARGET            
            # inv = target.split('Inventory:')[1]
            # inv = inv.split('\n')[0]
            # inv_str = inv.replace(' ', '')
            # obj_states_2 = []
            # obj_states_3 = []
            
            # objects = target.split('Information:')[1]
            # objects = objects.split('\n')
            # for obj in objects:
            #     obj = obj.split(')')[-1]
            #     obj_list = obj.split(',')
            #     for i in range(len(obj_list)):
            #         obj_list[i] = obj_list[i].replace(' ', '')
            #     if len(obj_list) == 3:
            #         obj_states_2.append(obj_list)
            #     elif len(obj_list) == 4: 
            #         obj_states_3.append(obj_list)

            #execute code
            async_function_regex = re.compile(r'async\s+function\s+(\w+)\s*\(\s*bot\s*\)\s*{')

            # 查找所有匹配的async函数并输出对应的await调用
            function_name = async_function_regex.findall(code_str)[0]
            exec_code = f"await {function_name}(bot);"        
            return {
                "explain": explain_str,
                "subtask": subtask_str,
                "code": code_str,
                "exec_code":exec_code
                # "inventory": inv_str,
                # "obj_2": obj_states_2, 
                # "obj_3": obj_states_3,
            }
        except Exception as e:
            error = e
        return f"Error parsing response (before program execution): {error}"

    # def process_ai_message(self, message):
    #     # assert isinstance(message, AIMessage)

    #     retry = 3
    #     error = None
    #     while retry > 0:
    #         try:
    #             babel = require("@babel/core")
    #             babel_generator = require("@babel/generator").default

    #             code_pattern = re.compile(r"```(?:javascript|js)(.*?)```", re.DOTALL)
    #             code = "\n".join(code_pattern.findall(message.content))
    #             parsed = babel.parse(code)
    #             functions = []
    #             assert len(list(parsed.program.body)) > 0, "No functions found"
    #             for i, node in enumerate(parsed.program.body):
    #                 if node.type != "FunctionDeclaration":
    #                     continue
    #                 node_type = (
    #                     "AsyncFunctionDeclaration"
    #                     if node["async"]
    #                     else "FunctionDeclaration"
    #                 )
    #                 functions.append(
    #                     {
    #                         "name": node.id.name,
    #                         "type": node_type,
    #                         "body": babel_generator(node).code,
    #                         "params": list(node["params"]),
    #                     }
    #                 )
    #             # find the last async function
    #             main_function = None
    #             for function in reversed(functions):
    #                 if function["type"] == "AsyncFunctionDeclaration":
    #                     main_function = function
    #                     break
    #             assert (
    #                 main_function is not None
    #             ), "No async function found. Your main function must be async."
    #             assert (
    #                 len(main_function["params"]) == 1
    #                 and main_function["params"][0].name == "bot"
    #             ), f"Main function {main_function['name']} must take a single argument named 'bot'"
    #             program_code = "\n\n".join(function["body"] for function in functions)
    #             exec_code = f"await {main_function['name']}(bot);"
    #             return {
    #                 "program_code": program_code,
    #                 "program_name": main_function["name"],
    #                 "exec_code": exec_code,
    #             }
    #         except Exception as e:
    #             retry -= 1
    #             error = e
    #             time.sleep(1)
    #     return f"Error parsing action response (before program execution): {error}"

    def summarize_chatlog(self, events):
        def filter_item(message: str):
            craft_pattern = r"I cannot make \w+ because I need: (.*)"
            craft_pattern2 = (
                r"I cannot make \w+ because there is no crafting table nearby"
            )
            mine_pattern = r"I need at least a (.*) to mine \w+!"
            if re.match(craft_pattern, message):
                return re.match(craft_pattern, message).groups()[0]
            elif re.match(craft_pattern2, message):
                return "a nearby crafting table"
            elif re.match(mine_pattern, message):
                return re.match(mine_pattern, message).groups()[0]
            else:
                return ""

        chatlog = set()
        for event_type, event in events:
            if event_type == "onChat":
                item = filter_item(event["onChat"])
                if item:
                    chatlog.add(item)
        return "I also need " + ", ".join(chatlog) + "." if chatlog else ""

class ActionAgent:
    def __init__(
        self,
        model_name="gpt-3.5-turbo",
        temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            print(f"\033[32mLoading Action Agent from {ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(f"{ckpt_dir}/action/chest_memory.json")
        else:
            self.chest_memory = {}
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )

    def update_chest_memory(self, chests):
        for position, chest in chests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    print(
                        f"\033[32mAction Agent removing chest {position}: {chest}\033[0m"
                    )
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    print(f"\033[32mAction Agent saving chest {position}: {chest}\033[0m")
                    self.chest_memory[position] = chest
        U.dump_json(self.chest_memory, f"{self.ckpt_dir}/action/chest_memory.json")

    def render_chest_observation(self):
        chests = []
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) > 0:
                chests.append(f"{chest_position}: {chest}")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) == 0:
                chests.append(f"{chest_position}: Empty")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, str):
                assert chest == "Unknown"
                chests.append(f"{chest_position}: Unknown items inside")
        assert len(chests) == len(self.chest_memory)
        if chests:
            chests = "\n".join(chests)
            return f"Chests:\n{chests}\n\n"
        else:
            return f"Chests: None\n\n"

    def render_system_message(self, skills=[]):
        system_template = load_prompt("action_template")
        # FIXME: Hardcoded control_primitives
        base_skills = [
            "exploreUntil",
            "mineBlock",
            "craftItem",
            "placeItem",
            "smeltItem",
            "killMob",
        ]
        if not self.llm.model_name == "gpt-3.5-turbo":
            base_skills += [
                "useChest",
                "mineflayer",
            ]
        programs = "\n\n".join(load_control_primitives_context(base_skills) + skills)
        response_format = load_prompt("action_response_format")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(
            programs=programs, response_format=response_format
        )
        assert isinstance(system_message, SystemMessage)
        return system_message

    def render_human_message(
        self, *, events, code="", task="", context="", critique=""
    ):
        chat_messages = []
        error_messages = []
        # FIXME: damage_messages is not used
        damage_messages = []
        assert events[-1][0] == "observe", "Last event must be observe"
        for i, (event_type, event) in enumerate(events):
            if event_type == "onChat":
                chat_messages.append(event["onChat"])
            elif event_type == "onError":
                error_messages.append(event["onError"])
            elif event_type == "onDamage":
                damage_messages.append(event["onDamage"])
            elif event_type == "observe":
                biome = event["status"]["biome"]
                time_of_day = event["status"]["timeOfDay"]
                voxels = event["voxels"]
                entities = event["status"]["entities"]
                health = event["status"]["health"]
                hunger = event["status"]["food"]
                position = event["status"]["position"]
                equipment = event["status"]["equipment"]
                inventory_used = event["status"]["inventoryUsed"]
                inventory = event["inventory"]
                assert i == len(events) - 1, "observe must be the last event"

        observation = ""

        if code:
            observation += f"Code from the last round:\n{code}\n\n"
        else:
            observation += f"Code from the last round: No code in the first round\n\n"

        if self.execution_error:
            if error_messages:
                error = "\n".join(error_messages)
                observation += f"Execution error:\n{error}\n\n"
            else:
                observation += f"Execution error: No error\n\n"

        if self.chat_log:
            if chat_messages:
                chat_log = "\n".join(chat_messages)
                observation += f"Chat log: {chat_log}\n\n"
            else:
                observation += f"Chat log: None\n\n"

        observation += f"Biome: {biome}\n\n"

        observation += f"Time: {time_of_day}\n\n"

        if voxels:
            observation += f"Nearby blocks: {', '.join(voxels)}\n\n"
        else:
            observation += f"Nearby blocks: None\n\n"

        if entities:
            nearby_entities = [
                k for k, v in sorted(entities.items(), key=lambda x: x[1])
            ]
            observation += f"Nearby entities (nearest to farthest): {', '.join(nearby_entities)}\n\n"
        else:
            observation += f"Nearby entities (nearest to farthest): None\n\n"

        observation += f"Health: {health:.1f}/20\n\n"

        observation += f"Hunger: {hunger:.1f}/20\n\n"

        observation += f"Position: x={position['x']:.1f}, y={position['y']:.1f}, z={position['z']:.1f}\n\n"

        observation += f"Equipment: {equipment}\n\n"

        if inventory:
            observation += f"Inventory ({inventory_used}/36): {inventory}\n\n"
        else:
            observation += f"Inventory ({inventory_used}/36): Empty\n\n"

        if not (
            task == "Place and deposit useless items into a chest"
            or task.startswith("Deposit useless items into the chest at")
        ):
            observation += self.render_chest_observation()

        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        if critique:
            observation += f"Critique: {critique}\n\n"
        else:
            observation += f"Critique: None\n\n"

        return HumanMessage(content=observation)

    def process_ai_message(self, message):
        assert isinstance(message, AIMessage)

        retry = 3
        error = None
        while retry > 0:
            try:
                babel = require("@babel/core")
                babel_generator = require("@babel/generator").default

                code_pattern = re.compile(r"```(?:javascript|js)(.*?)```", re.DOTALL)
                code = "\n".join(code_pattern.findall(message.content))
                parsed = babel.parse(code)
                functions = []
                assert len(list(parsed.program.body)) > 0, "No functions found"
                for i, node in enumerate(parsed.program.body):
                    if node.type != "FunctionDeclaration":
                        continue
                    node_type = (
                        "AsyncFunctionDeclaration"
                        if node["async"]
                        else "FunctionDeclaration"
                    )
                    functions.append(
                        {
                            "name": node.id.name,
                            "type": node_type,
                            "body": babel_generator(node).code,
                            "params": list(node["params"]),
                        }
                    )
                # find the last async function
                main_function = None
                for function in reversed(functions):
                    if function["type"] == "AsyncFunctionDeclaration":
                        main_function = function
                        break
                assert (
                    main_function is not None
                ), "No async function found. Your main function must be async."
                assert (
                    len(main_function["params"]) == 1
                    and main_function["params"][0].name == "bot"
                ), f"Main function {main_function['name']} must take a single argument named 'bot'"
                program_code = "\n\n".join(function["body"] for function in functions)
                exec_code = f"await {main_function['name']}(bot);"
                return {
                    "program_code": program_code,
                    "program_name": main_function["name"],
                    "exec_code": exec_code,
                }
            except Exception as e:
                retry -= 1
                error = e
                time.sleep(1)
        return f"Error parsing action response (before program execution): {error}"

    def summarize_chatlog(self, events):
        def filter_item(message: str):
            craft_pattern = r"I cannot make \w+ because I need: (.*)"
            craft_pattern2 = (
                r"I cannot make \w+ because there is no crafting table nearby"
            )
            mine_pattern = r"I need at least a (.*) to mine \w+!"
            if re.match(craft_pattern, message):
                return re.match(craft_pattern, message).groups()[0]
            elif re.match(craft_pattern2, message):
                return "a nearby crafting table"
            elif re.match(mine_pattern, message):
                return re.match(mine_pattern, message).groups()[0]
            else:
                return ""

        chatlog = set()
        for event_type, event in events:
            if event_type == "onChat":
                item = filter_item(event["onChat"])
                if item:
                    chatlog.add(item)
        return "I also need " + ", ".join(chatlog) + "." if chatlog else ""
