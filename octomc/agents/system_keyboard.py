import time
import pyautogui

# Wait some time to ensure the switch to the Minecraft window
def rotate():
    yaws = [0, 90, 180, 270]
    for yaw in yaws:
        pyautogui.press('t')  # Press 't' to open chat

        # Press backspace twice (if needed to clear previous content)
        pyautogui.press('backspace')
        pyautogui.press('backspace')
        command = f"/execute as bot at @s run tp @s ~ ~ ~ {yaw} 0"
        pyautogui.write(command)
        time.sleep(1)  # Wait a moment for the command to register
        pyautogui.press('enter')  # Send the command

def change_to_bot():
    time.sleep(0.5)  # Short wait

    # Press 't' to open chat
    pyautogui.press('t')

    # Press backspace twice (if needed to clear previous content)
    pyautogui.press('backspace')
    pyautogui.press('backspace')

    # Send /spectate bot command
    command = "/spectate bot"
    pyautogui.write(command)
    time.sleep(1)  # Wait a moment for the command to register
    pyautogui.press('enter')  # Send the command

def capture():
    time.sleep(0.5)  # Short wait
    pyautogui.press('f2')  # Press F2 to take a screenshot

def change_gamemode(mode):
    time.sleep(1)  # Short wait

    # Press 't' to open chat
    pyautogui.press('t')

    # Press backspace twice (if needed to clear previous content)
    pyautogui.press('backspace')
    pyautogui.press('backspace')

    # Send change gamemode command
    if mode == "spectator":
        command = "/gamemode spectator"
    else:
        command = "/gamemode creative"
    pyautogui.write(command)
    time.sleep(1)  # Wait a moment for the command to register
    pyautogui.press('enter')  # Send the command

if __name__ == "__main__":
    while True:
        change_to_bot()
        change_gamemode("spectator")
        capture()
        change_to_bot()
        change_gamemode("creative")
        capture()
