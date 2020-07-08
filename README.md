# Hateful Memes Example using MMF

* Kiela, D., Firooz, H., Mohan A., Goswami, V., Singh, A., Ringshia P. & Testuggine, D. (2020). *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes*. arXiv preprint arXiv:2005.04790

```
@article{kiela2020hateful,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and  Testuggine, Davide},
  journal={arXiv preprint arXiv:2005.04790},
  year={2020}
}
```
* [Citation for MMF](https://github.com/facebookresearch/mmf/tree/master/README.md#citation)

Links: [[arxiv]](https://arxiv.org/abs/2005.04790) [[challenge]](https://www.drivendata.org/competitions/64/hateful-memes/) [[blog post]](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set)

The example tries to replicate the model developed in DrivenData's [blog post](https://www.drivendata.co/blog/hateful-memes-benchmark/) on the Hateful Memes.

## Installation

Preferably, you would create your own conda environment like the following:

```
git clone git@github.com:holman57/Hateful-Memes.git
cd Hateful-Memes
conda create --name mmf
conda activate mmf
conda install -c anaconda pip
pip install -r requirements.txt
```

## Prerequisites

Please follow prerequisites for the Hateful Memes dataset at [this link](https://fb.me/hm_prerequisites).

## Running

Run training with the following command on the Hateful Memes dataset:

```
MMF_USER_DIR="." mmf_run config="configs/experiments/defaults.yaml"  model=concat_vl dataset=hateful_memes training.num_workers=0
```

We set `training.num_workers=0` here to avoid memory leaks with fasttext.
Please follow [configuration](https://mmf.readthedocs.io/en/latest/notes/configuration_system.html) document to understand how to use MMF's configuration system to update parameters.

## Directory Structure

```
├── configs
│   ├── experiments
│   │   └── defaults.yaml
│   └── models
│       └── concat_vl.yaml
├── __init__.py
├── models
│   ├── concat_vl.py
├── processors
│   ├── processors.py
├── README.md
└── requirements.txt
```

Some notes:

1. Configs have been divided into `experiments` and `models` where experiments will contain training configs while models will contain model specific config we implmented.
2. `__init__.py` imports all of the relevant files so that MMF can find them. This is what `env.user_dir` actually looks for.
3. `models` directory contains our model implementation, in this case specifically `concat_vl`.
4. `processors` contains our project specific processors implementation, in this case, we implemented FastText processor for Sentence Vectors.


