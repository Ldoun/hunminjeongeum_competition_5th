# nsml: ldoun8260/nsml:latest

from distutils.core import setup

setup(
    name="ladder_networks",
    version="1.0",
    install_requires=[
        "jamo",
        "numpy==1.20.1",
        "jamotools",
        "librosa",
        "pandas",
        "tensorboard==2.4.1",
        "tensorboard-plugin-wit==1.8.0",
        "tensorboardX==2.1",
        "torch-optimizer",
        "torch==1.7.1",
        "torchmetrics",
        "tqdm",
        "transformers",
        "apex",
        "datasets",
        "jiwer",
        "ctcdecode"
    ],
)
