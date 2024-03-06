from voyager_exp import Voyager

import openai

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
azure_login = {
    "client_id": "91c66194-9999-442e-a289-777ca1202bfa",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "None",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}
from agents.choiszt_keyboard import change_gamemode
# change_gamemode("spectator")
voyager = Voyager(
    mc_port=35353,
    openai_api_key=openai.api_key,
)

# start lifelong learning
task="Kill 2 pigs"

# Kill a llama
# Kill 2 pigs

voyager.capture(task)

