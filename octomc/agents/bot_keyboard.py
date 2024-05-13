import time
import requests
import json

def rotate(send,programs):
    yaws=[0,90,180,270]
    for yaw in yaws:
        command=f"bot.chat('/execute as bot at @s run tp @s ~ ~ ~ {yaw} 0');"
        send(command,programs=programs)
        # time.sleep(1)

def change_to_bot(send,programs):
    command = "bot.chat('/spectate bot')" #dont work -> use keyboard input
    send(command,programs=programs)

def capture(send,programs):
    with open("./basic_move/explore.js","r")as f:
        code=f.read()
    return json.loads((send(code,programs)).json())

def initial(send,programs):
    # command=f"bot.chat('/execute as bot at @s run tp @s -38627 77 36953');" #stone
    # command=f"bot.chat('/execute as bot at @s run tp @s 3267 66 -2828');" #dirt
    # command=f"bot.chat('/execute as bot at @s run tp @s 3355 71 -3207');" #beach
    # command=f"bot.chat('/execute as bot at @s run tp @s 5849 88 -820');" #jungle
    # command=f"bot.chat('/execute as bot at @s run tp @s -91 66 563');" #forest
    # command=f"bot.chat('/execute as bot at @s run tp @s -321 74 545');" #village
    # command=f"bot.chat('/execute as bot at @s run tp @s 5195 77 -4468');" #village    
    # command=f"bot.chat('/execute as bot at @s run tp @s 5271 73 -4302');" #village 
    command=f"bot.chat('/execute as bot at @s run tp @s 5305 64 -4549');"       
    send(command,programs=programs)

def move(send,programs):
    with open("./basic_move/move.js","r")as f:
        code=f.read()
    return json.loads((send(code,programs)).json())

def preserve(send,programs):
    info=json.loads(send('',programs).json())
    return info

def see(send,programs):
    command = "bot.blockInSight();" 
    return json.loads(send(command,programs=programs).json())

def look_around(send,programs):
    with open("./basic_move/look_around.js","r")as f:
        code=f.read()
    return json.loads((send(code,programs)).json())

def see(send,programs):
    with open("./basic_move/see.js","r")as f:
        code=f.read()
    return json.loads((send(code,programs)).json())