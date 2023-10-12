<p align="center" width="100%">
<img src="https://i.mji.rip/2023/10/13/c4bdc6505f2b3f2304bffb5ea196f5a2.png"  width="40%" height="80%">
</p>
<div>
<div align="center">
<font size=5><strong>Octopus: Embodied Vision-Language Programmer from Environmental Feedback</strong></font>
<br>
    <a href='https://jingkang50.github.io/' target='_blank'>Jingkang Yang<sup>*,1</sup></a>&emsp;
    <a href='https://github.com/dongyh20/' target='_blank'>Yuhao Dong<sup>*,2,5</sup></a>&emsp;
    <a href='https://github.com/choiszt/' target='_blank'>Shuai Liu<sup>*,3,5</sup></a>&emsp;
    <a href='https://brianboli.com/' target='_blank'>Bo Li<sup>*,1</sup></a>&emsp;
    </br>
    Ziyue Wang<sup>&dagger;,1</sup></a>&emsp;
	Chencheng Jiang<sup>&dagger;,4</sup></a>&emsp;
    Haoran Tan<sup>&dagger;,3</sup></a>&emsp;
    Jiamu Kang<sup>&dagger;,2</sup></a>&emsp;
	</br>
    <a href='https://zhangyuanhan-ai.github.io/' target='_blank'>Yuanhan Zhang<sup>1</sup></a>&emsp;
	<a href='https://kaiyangzhou.github.io/' target='_blank'>Kaiyang Zhou<sup>1</sup></a>&emsp;
	<a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>1,5,&#x2709</sup></a>
</div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    <sup>2</sup>Tsinghua University&emsp;
    <br>
    <sup>3</sup>Beijing University of Posts and Telecommunications&emsp;&emsp;
    <br>
    <sup>4</sup>Xi'an Jiaotong University&emsp;
    <sup>5</sup>Shanghai AI Laboratory&emsp;
    </br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>&dagger;</sup> Equal Engineering Contribution&emsp;
    <sup>&#x2709</sup> Corresponding Author
</div>


-----------------

![](https://img.shields.io/badge/octopus-v0.1-darkcyan)
![](https://img.shields.io/github/stars/dongyh20/Octopus?style=social)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fdongyh20%2FOctopus&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
![](https://img.shields.io/badge/code%20style-black-000000.svg)

[Project Page](https://choiszt.github.io/Octopus) | [Octopus Paper](https://arxiv.org/)

## 🐙 Introducing Octopus
Octopus is a novel VLM designed to proficiently decipher an agent’s vision and textual task objectives and to formulate intricate action sequences and generate executable code.

This repository provides:
- Training data collection pipeline in `octogibson` environment,
- Evaluation pipeline in `octogibson` environment,
- Evaluation pipeline in `octogta` environment,
- Training pipeline of the `octopus` model.

**Contact: Leave issue or contact `jingkang001@e.ntu.edu.sg` and `dongyh20@mails.tsinghua.edu.cn`. We are on call to respond.**

## 🦾 Updates

**[2023-10]**

1. 🤗 Introducing Project Octopus' homepage: https://choiszt.github.io/Octopus.
2. 🤗 Check our [paper](https://arxiv.org/abs/???) introducing Octopus in details. 


## 🏁 Get Started
1. **Training Data Collection:** For data collection from `octogibson` environment, you need to set up two conda environments: `omnigibson` and `gpt4`. The `omnigibson` environment has an agent to act following the instruction from `gpt4` environment. Please checkout [here](octogibson/README.md) for detailed information.
2. **Evaluation in OctoGibson:** We provide the pipeline that the simulator sends messages to the Octopus server and gets responses to control the agent.
3. **Evaluation in OctoGTA:** We provide instructions, code, and MOD so that the Octopus can complete tasks in the GTA environment. Please checkout [here](octogta/README.md) for detailed information.
4. **Octopus Training:** We provide code for training Octopus. Please checkout [here](octopus/README.md) for detailed information.


## 📑 Citation

If you found this repository useful, please consider citing:
```
@article{yang2023octopus,
    author = {Jingkang Yang and Yuhao Dong and Shuai Liu and Bo Li and Ziyue Wang and Chencheng Jiang and Haoran Tan and Jiamu Kang and Yuanhan Zhang and Kaiyang Zhou and Ziwei Liu},
    title = {Octopus: Embodied Vision-Language Programmer from Environmental Feedback},
    year = {2023},
    license = {\url{http://arxiv.org/licenses/nonexclusive-distrib/1.0/}}
}
```

### 👨‍🏫 Acknowledgements

We thank the [OmniGibson](https://github.com/StanfordVL/OmniGibson) team for their help and great contribution to the open-source community.
