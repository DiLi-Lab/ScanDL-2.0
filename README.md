# ScanDL 2.0: A Generative Model of Eye Movements in Reading Synthesizing Scanpaths and Fixation Durations

This repository contains the code to reproduce the experiments in [ScanDL 2.0: A Generative Model of Eye Movements in Reading Synthesizing Scanpaths and Fixation Durations](https://doi.org/10.1145/3725830). 

Moreover, it contains pre-trained weights for both a paragraph-level and a sentence-level version ScanDL 2.0 that can be used to generate scanpaths without requiring training from scratch.


## Setup 

### Clone repository 
```bash
git clone https://github.com/DiLi-Lab/ScanDL-2.0.git
cd ScanDL-2.0
```


### Install requirements
The code is based on the PyTorch and huggingface modules.
```bash
pip install -r requirements.txt
```


## Using pre-trained ScanDL 2.0

We pre-trained ScanDL 2.0 on EMTeC for a paragraph-level version of ScanDL 2.0 (generates scanpaths on paragraphs) and on CELER for a sentence-level version of ScanDL 2.0 (generates scanpaths on single sentences). 

To use the weights, download the `models.zip` file in the latest release. Unzip it and ensure that the paths in `scandl2_pkg/PATHS.py` are correct.

An example of how ScanDL 2.0 can be used (i.e., what the input and output is and how it is loaded) is provided in `scandl2_pkg/test.py`. For inference, you need one GPU set up with [CUDA](https://developer.nvidia.com/cuda-toolkit). 

Simply run:
```bash
CUDA_VISIBLE_DEVICES=0 python -m scandl_pkg.test
```

When loading pre-trained ScanDL 2.0, the arguments are as follows:
```python
class ScanDL2(
    text_type: str = sentence,
    bsz: Optional[int] = 2,
    save: Optional[str] = None,
    filename: Optional[str] = None,
    )
```
Parameters:
* `text_type` (str): either 'sentence' or 'paragraph', defaults to 'sentence'
* `bsz` (int): the batch size used for the inference (depends on GPU capacity): defaults to 2
* `save` (str): if provided, the model output will be automatically saved as a `json` file at the provided path. Defaults to `None` (output is not automatically saved)
* `filename`(str): if provided, the output is saved in a file with the given name. If `save` is given but `filename` is not, `filename` defaults to `scandl2_outputs.json`.

Input:
* A single `str` or `List[str]`


Returns:
* an object of type 

    ```python 
    Dict[str, Union[List[List[str]], List[List[int]], List[str], List[List[float]]]]
    ```
    containing the 
    * predicted scanpath in words
    * predicted scanpath in ids
    * the original sentence as list of words
    * the predicted fixation durations
    * a unique idx for every input sentence



## Reproducing the training, inference and evaluation of [ScanDL 2.0: A Generative Model of Eye Movements in Reading Synthesizing Scanpaths and Fixation Durations](https://doi.org/10.1145/3725830). 

### Preliminaries

#### Download the data

The **CELER** data can be downloaded from this [link](https://github.com/berzak/celer), where you need to follow the description.

The **ZuCo** data can be downloaded from this [OSF repository](https://osf.io/q3zws/). You can use `diffusion_only/scripts/get_zuco_data.sh` to automatically download the ZuCo data. Note, ZuCo is a big dataset and requires a lot of storage.

The **Beijing Sentence Corpus (BSC)** can be downloaded from this [OSF repository](https://osf.io/vr3k8/).

The **Eye Movements on Machine-Generated Texts Corpus (EMTeC)** can be downloaded from this [OSF repository](https://osf.io/ajqze/) or using [this Python script](https://github.com/DiLi-Lab/EMTeC/blob/main/get_et_data.py).

Make sure you adapt the path to the folder that contains ```celer```, ```zuco```, ```bsc``` and ```emtec``` in the file ```CONSTANTS.py```. If you use aboves bash script `diffusion_only/scripts/get_zuco_data.sh`, the `zuco` path is `data/`.
Make sure there are no whitespaces in the zuco directories (there might be when you download the data). You might want to check the script ```diffusion_only/scripts/sp_load_celer_zuco.load_zuco()``` for the spelling of the directories.


#### Pre-process the training and test data


Preprocessing the eye-tracking data takes time. It is thus recommended to perform the preprocessing once for each setting and save the preprocessed data in the directories ``processed_data``, ``processed_data_emtec``, and ``processed_data_bsc``.
This not only saves time if training is performed several times but it also ensures the same data splits for each training run in the same setting.

Make sure you have adapted the paths to your datasets in `CONSTANTS.py` 
For preprocessing and saving the data, run
```bash
python create_data_splits.py
python create_data_splits.py --emtec
python_create_data_splits.py --bsc 
```


### ScanDL 2.0

To execute the training and inference commands, you need GPUs set up with [CUDA](https://developer.nvidia.com/cuda-toolkit).

#### ScanDL Module


Train the ScanDL Module for fixation location prediction in all evaluation settings and for all datasets. You might want to adapt your available GPUs in `CUDA_VISIBLE_DEVICES`. 

Adapt the paths where you want to save the model checkpoints (`SCANDL_MODULE_TRAIN_PATH`, `SCANDL_MODULE_TRAIN_PATH_EMTEC`, `SCANDL_MODULE_TRAIN_PATH_BSC`) and where you want to save the inference outputs (`SCANDL_MODULE_INF_PATH`, `SCANDL_MODULE_INF_PATH_EMTEC`, `SCANDL_MODULE_INF_PATH_BSC`) in `CONSTANTS.py`.

To train the ScanDL Modules, please run the following bash scripts.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash train_scandl_module.sh
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash train_scandl_module_emtec.sh
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash train_scandl_module_bsc.sh
```

To run the inference on the ScanDL Modules, please run the following bash scripts.
```bash
CUDA_VISIBLE_DEVICES=0 bash inf_scandl_module.sh
```
```bash
CUDA_VISIBLE_DEVICES=0 bash inf_scandl_module_emtec.sh
```
```bash
CUDA_VISIBLE_DEVICES=0 bash inf_scandl_module_bsc.sh
```

#### Fixation Duration Module

Adapt the paths where you want to save the model checkpoints (`FIXDUR_MODULE_TRAIN_PATH`, `FIXDUR_MODULE_TRAIN_PATH_EMTEC`, `FIXDUR_MODULE_TRAIN_PATH_BSC`) and where you want to save the inference outputs (`FIXDUR_MODULE_INF_PATH`, `FIXDUR_MODULE_INF_PATH_EMTEC`, `FIXDUR_MODULE_INF_PATH_BSC`) in `CONSTANTS.py`.

To train the Fixation Duration Modules, please run the following bash scripts.
```bash
CUDA_VISIBLE_DEVICES=0 bash train_fixdur_module.sh
```
```bash
CUDA_VISIBLE_DEVICES=0 bash train_fixdur_module_emtec.sh
```
```bash
CUDA_VISIBLE_DEVICES=0 bash train_fixdur_module_bsc.sh
```

The run the inference on the Fixation Duration modules, please run the following bash script.
```bash
CUDA_VISIBLE_DEVICES=0 bash inf_fixdur_module.sh
```
```bash
CUDA_VISIBLE_DEVICES=0 bash inf_fixdur_module_emtec.sh
```
```bash
CUDA_VISIBLE_DEVICES=0 bash inf_fixdur_module_bsc.sh
```

#### Evaluation

To evaluate ScanDL fix-dur, call the following scripts. This will further also run the evaluation on the Human baseline.
```bash
python -m scandl_fixdur.fix_dur_module.sp_eval_seq2seq
```
```bash
python -m scandl_fixdur.fix_dur_module.sp_eval_seq2seq --setting reader combined --emtec 
```
```bash
python -m scandl_fixdur.fix_dur_module.sp_eval_seq2seq --setting reader combined --bsc 
```





### Ablation: ScanDL diff-dur

Adapt the paths where you want to save the model checkpoints (`DIFFUSION_ONLY_TRAIN_PATH`) and where you want to save the inference outputs (`DIFFUSION_ONLY_INF_PATH`) in `CONSTANTS.py`.

To train the diffusion-only architecture on the first fold of the _New Reader_ setting, please run the following bash script.
#### Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash train_diffusion_only.sh
```

To run the inference of the diffusion-only architecture, please run the following bash script.
#### Inference 
```bash
CUDA_VISIBLE_DEVICES=0 bash inf_diffusion_only.sh
    
```

#### Evaluation
To evaluate the performance of the diffusion-only architecture, please run the following bash script.
```bash
python -m diffusion_only.scripts.sp_eval_hp
```



### Training the paragraph-level and sentence-level versions of ScanDL 2.0

#### pre-process the data

```bash
python -m scandl2_pkg.create_data --data emtec
python -m scandl2_pkg.create_data --data celer
```


#### train ScanDL module 

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash run_train_complete_scandl_module.sh
```

#### train the Fixation Duration module

```bash
CUDA_VISIBLE_DEVICES=0 bash run_train_complete_fixdur_module.sh
```



## Citation

```bibtex
@article{bolliger2025scandl2,
	author = {Bolliger, Lena S. and Reich, David R. and J\"{a}ger, Lena A.},
	title = {ScanDL 2.0: A Generative Model of Eye Movements in Reading Synthesizing Scanpaths and Fixation Durations},
	year = {2025},
	issue_date = {May 2025},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	volume = {9},
	number = {ETRA5},
	url = {https://doi.org/10.1145/3725830},
	doi = {0.1145/3725830},
	abstract = {Eye movements in reading have become a vital tool for investigating the cognitive mechanisms involved in language processing. They are not only used within psycholinguistics but have also been leveraged within the field of NLP to improve the performance of language models on downstream tasks. However, the scarcity and limited generalizability of real eye-tracking data present challenges for data-driven approaches. In response, synthetic scanpaths have emerged as a promising alternative. Despite advances, however, existing machine learning-based methods, including the state-of-the-art ScanDL (Bolliger et al. 2023), fail to incorporate fixation durations into the generated scanpaths, which are crucial for a complete representation of reading behavior. We therefore propose a novel model, denoted ScanDL 2.0, which synthesizes both fixation locations and durations. It sets a new benchmark in generating human-like synthetic scanpaths, demonstrating superior performance across various evaluation settings. Furthermore, psycholinguistic analyses confirm its ability to emulate key phenomena in human reading. Our code as well as pre-trained model weights are available via https://github.com/DiLi-Lab/ScanDL-2.0.},
	journal = {Proceedings of the ACM on Human-Computer Interaction},
	month = may,
	articleno = {5},
	numpages = {21},
	keywords = {neural networks, scanpath generation, eye movements, reading, diffusion models}
}
```