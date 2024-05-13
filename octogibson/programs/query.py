import re
import time
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import utils.gpt_utils as gu


class Query:
    def __init__(self):
        self.history_info = {}
        self.record_history()

    def render_system_message(self):
        system_template = gu.load_prompt("prompt_template")

        response_format = gu.load_prompt("response_template")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(response_format=response_format)
        assert isinstance(system_message, SystemMessage)
        return system_message
    
    def render_human_message(self, human_info):
        scene_graph, object, inventory, task = human_info
        message = ""    
        message += f"Observed Objects: {object}\n"
        message += f"Observed Relations: {scene_graph}\n"
        if inventory=="[]":
            message += f"Inventory: None\n"
        elif inventory:
            message += f"Inventory: {inventory[2:-2]}\n"

        message += f"Task Goal: {task}\n"
        
        if len(self.history_info['subtask']) > 0:
            message += f"Original Subtasks: {self.history_info['subtask']}\n"
        else:
            message += f"Original Subtasks: None\n"

        if len(self.history_info['code']) > 0:
            message += f"Previous Action Code: {self.history_info['code']}\n"
            if len(self.history_info['error']) > 0:
                message += f"Execution Error: {self.history_info['error']}\n"
            else:
                message += f"Execution Error: No error\n"  
        elif len(self.history_info['code']) == 0: 
            message += f"Previous Action Code: No code\n"
            message += f"Execution error: No error\n"  
        
        message += "Now, please output Explain, Subtasks (revise if necessary), Code that completing the next subtask, and Target States, according to the instruction above. Remember you can only use the functions provided above and pay attention to the response format."
            
        return HumanMessage(content=message)
    
    def process_ai_message(self, processed_message):
        retry = 1
        error = None
        classes = ["Explain:", "Subtasks:", "Code:", "Target States:"]
        idxs = []
        for c in classes:
            m = processed_message.find(c)
            idxs.append(m)
        if -1 in idxs:
            raise Exception('Invalid response format!')
        
        # parse process
        while retry > 0:
            try:
                explain = processed_message[:idxs[1]]
                subtask = processed_message[idxs[1]:idxs[2]]
                code = processed_message[idxs[2]:idxs[3]]
                target = processed_message[idxs[3]:]
                
                #EXPLAIN
                explain_str = explain.split('Explain:')[1]
                explain_str = explain_str.replace('\n', '')
                explain_str = explain_str.replace('\n\n', '')
                
                #SUBTASK
                subtask_str = subtask.split('Subtasks:')[1]
                subtask_str = subtask_str.replace('\n\n', '')
                
                #CODE
                code_str = code.split('```python\n')[1].split('```')[0]
                
                #TARGET            
                inv = target.split('Inventory:')[1]
                inv = inv.split('\n')[0]
                inv_str = inv.replace(' ', '')
                obj_states_2 = []
                obj_states_3 = []
                
                objects = target.split('Information:')[1]
                objects = objects.split('\n')
                for obj in objects:
                    obj = obj.split(')')[-1]
                    obj_list = obj.split(',')
                    for i in range(len(obj_list)):
                        obj_list[i] = obj_list[i].replace(' ', '')
                    if len(obj_list) == 3:
                        obj_states_2.append(obj_list)
                    elif len(obj_list) == 4: 
                        obj_states_3.append(obj_list)
                        
                return {
                    "explain": explain_str,
                    "subtask": subtask_str,
                    "code": code_str,
                    "inventory": inv_str,
                    "obj_2": obj_states_2, 
                    "obj_3": obj_states_3,
                }
            except Exception as e:
                retry -= 1
                error = e
        return f"Error parsing response (before program execution): {error}"
    
    def record_history(self, subtask="", code="", error=""):
        self.history_info['subtask'] = subtask
        self.history_info['code'] = code
        self.history_info['error'] = error
