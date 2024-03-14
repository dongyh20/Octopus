import time
import pyautogui

# 等待一些时间以确保切换到Minecraft窗口
def rotate():
    yaws=[0,90,180,270]
    for yaw in yaws:
        pyautogui.press('t')

    # 发送两次Backspace（如果需要清除之前的内容）
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        command=f"/execute as bot at @s run tp @s ~ ~ ~ {yaw} 0"
        pyautogui.write(command)
        time.sleep(1)
        pyautogui.press('enter')
def change_to_bot():
    time.sleep(0.5)

    # 发送按键 "t" 打开聊天框
    pyautogui.press('t')

    # 发送两次Backspace（如果需要清除之前的内容）
    pyautogui.press('backspace')
    pyautogui.press('backspace')

    # 发送/spectate bot命令
    command = "/spectate bot"
    pyautogui.write(command)
    time.sleep(1)
    pyautogui.press('enter')
def capture():
    time.sleep
    pyautogui.press('f2')
def change_gamemode(mode):
    time.sleep(1)

    # 发送按键 "t" 打开聊天框
    pyautogui.press('t')

    # 发送两次Backspace（如果需要清除之前的内容）
    pyautogui.press('backspace')
    pyautogui.press('backspace')

    # 发送/spectate bot命令
    if mode=="spectator":
        command = "/gamemode spectator"
    else:
        command="/gamemode creative"
    pyautogui.write(command)
    time.sleep(1)
    pyautogui.press('enter')

if __name__=="__main__":
    while True:
        change_to_bot()
        change_gamemode("spectator")
        capture()
        change_to_bot()
        change_gamemode("creative")
        capture()
