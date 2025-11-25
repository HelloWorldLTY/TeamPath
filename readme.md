# TeamPath 🤝
This is the official code repo for the paper: Building MultiModal Pathology Experts with Reasoning AI Copilots.

## Installation

The most important thing is to install verl. Please install the platform directly from our folder **rl_train**. We have specified the version of verl. You can also go to check `2bd291e5494db03ba358ef279a334c2f0829b979` of the verl codebase for the best compatibility.

To install Llama factory for supervised fine-tuning, please refer this [link](https://github.com/hiyouga/LLaMA-Factory).

We also upload a conda yml file to reproduce our environment used for this project. We recommend users deploying this project in High Performance Computing centers. In personal computer, its installation time should be less than 10 minutes if you do not consider flash attention. To install it from conda, you can use:

```
conda env update -f verl.yml
```

The installation process in a personal computer will be 10 minutes, but we highly recommend using a High Performance Computer (HPC) for experiments.

## Accessing weights

The weights will be released at [here](https://huggingface.co/Pathstudy) after peer review.

## Training

Please refer our folder with names **rl_train** and **sft** for more details. We implement the framework with modified [verl](https://github.com/volcengine/verl) and [Llama-factory](https://github.com/hiyouga/LLaMA-Factory/tree/main), respectively. Moreover, we also support [trl](https://github.com/huggingface/trl/tree/main) and the codes will be provided upon request.

Please do not use the original huggingface token provided in this repo, which is expired. You need to replace it by your own token.

## Tutorials

We have provided several demo notebooks, which can be found in folder **clean_tutorial**


## Acknowledgement

We thank the developers of [verl](https://github.com/volcengine/verl), [Llama-factory](https://github.com/hiyouga/LLaMA-Factory/tree/main), and [trl](https://github.com/huggingface/trl/tree/main). We also thank [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) and authors of [PathGen](https://arxiv.org/abs/2407.00203), [HEST-1K](https://github.com/mahmoodlab/HEST), [STImage1k4M](https://github.com/JiawenChenn/STimage-1K4M), and [Patho-R1](https://github.com/wenchuan-zhang/patho-r1) for data and base model construction.

## Citation
```
@misc{liu2025teampathbuildingmultimodalpathology,
      title={TeamPath: Building MultiModal Pathology Experts with Reasoning AI Copilots}, 
      author={Tianyu Liu and Weihao Xuan and Hao Wu and Peter Humphrey and Marcello DiStasio and Heli Qi and Rui Yang and Simeng Han and Tinglin Huang and Fang Wu and Nan Liu and Irene Li and Hua Xu and Hongyu Zhao},
      year={2025},
      eprint={2511.17652},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2511.17652}, 
}
```



