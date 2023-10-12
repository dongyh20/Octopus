import os
import openai
openai.api_type = "azure"
openai.api_base = "/URL/FOR/YOUR/API/BASE"
openai.api_version = "YOUR API VERSION"
openai.api_key = "YOUR API KEY"

def gpt_request(content):
    response = openai.ChatCompletion.create(
        engine="YOUR ENGINE NAME",
        messages = [{"role":"user","content":content}],
        temperature=0,
        max_tokens=8000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    # print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']