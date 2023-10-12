# OctoGibson

## 🏁 OmniGibson Setup

You can follow the [OmniGibson Documentation]([Installation - OmniGibson Documentation (stanford.edu)](https://behavior.stanford.edu/omnigibson/getting_started/installation.html)) to install the OmniGibson. We have successfully installed OmniGibson by installing from source. 

After you successfully install the OmniGibson, run the following code to download all the assets needed for the simulation:

```python
cd OmniGibson
python scripts/download_datasets.py
```

Now, you may use the scripts below to try a simple demo provided by OmniGibson to test whether the installation is correct:

```python
python -m omnigibson.examples.scenes.scene_selector 
```

## 🏁 GPT-4 API Setup

Note that our experiments are conducted with the **GPT-4 32k** API provided by Azure OpenAI. As requirements are needed for GPT-4 as well as GPT-4 32k API, we strongly suggest you follow the [Azure Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-studio) to get your own GPT-4 32k deployment.

After deploying your own GPT-4 32k API model, you may test your deployment with the code:

```python
import os
import openai
openai.api_type = "azure"
openai.api_base = "/URL/FOR/YOUR/API/BASE" # replace it according to your deployment
openai.api_version = "YOUR API VERSION" # replace it according to your deployment
openai.api_key = "YOUR API KEY" # replace it according to your deployment

response = openai.ChatCompletion.create(
    engine="YOUR ENGINE NAME",
    messages = [{"role":"user","content":"Hello, I'm Octopus."}],
    temperature=0,
    max_tokens=8000,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)
print(response['choices'][0]['message']['content'])
```


## 📑 Citation

If you found this repository useful, please consider citing:
```
???
```
