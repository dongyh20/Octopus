# #Note: The openai-python library support for Azure OpenAI is in preview.
# import os
# import openai
# openai.api_type = "azure"
# openai.api_base = "https://voyager.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
# openai.api_key = "xxxx"

# def gpt_request(content):
#     response = openai.ChatCompletion.create(
#         engine="voyager",
#         messages = [{"role":"user","content":content}],
#         temperature=0.7,
#         max_tokens=800,
#         top_p=0.95,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=None
#         )
#     print(response['choices'][0]['message']['content'])
#     return response['choices'][0]['message']['content']

