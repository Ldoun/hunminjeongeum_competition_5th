# nsml: ldoun8260/nsml_stt2:1.0
#ldoun8260/nsml_stt2:1.0
#ldoun8260/nsml_stt1:1.0
#ldoun8260/nsml:9.0
#ldoun8260/nsml:12.0

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
        "torch",
        "torchmetrics",
        "tqdm",
        "transformers",
        "apex",
        "datasets",
        "jiwer==2.2.1",
        "pydub",
        "pyctcdecode==0.1.1",
        "gdown"
        #"torchaudio"
    ],
)
