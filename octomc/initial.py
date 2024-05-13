import json
def capture(send,programs):
    code='await look_around()'
    # with open("./basic_move/gpt_pipeline.js","r")as f:
    #     code=f.read()
    return json.loads((send(code,programs)).json())