#!/usr/bin/env python3
from shutil import ExecError
import json
import numpy as np
import torch
import torch.nn as nn
import time
from packaging import version
import math
from dataloader import *
from glob import glob
import os
import pandas as pd
import pickle
from datasets import load_metric
import argparse
import nsml
from nsml import HAS_DATASET, DATASET_PATH
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from apex import amp
from ctcdecode import CTCBeamDecoder
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence

def load(dir_name):
    save_dir = os.path.join(dir_name, "checkpoint")
    
    checkpoint = torch.load(save_dir)
    
    with open(os.path.join(dir_name, "dict_for_infer"), "rb") as f:
        dict_for_infer = pickle.load(f)

    return checkpoint, dict_for_infer
   

if __name__ == '__main__':
    model = Wav2Vec2ForCTC.from_pretrained("nuod/wav2vec2")
    model.freeze_feature_extractor()
    
    checkpoint, dict_for_infer = load('./nsml_model/nia1016_final_stt_3_131/582525/model')
    
    tokenizer = dict_for_infer["tokenizer"]
    model.lm_head = nn.Linear(
        in_features=768, out_features=len(tokenizer.txt2idx), bias=True
    )
    model.config = Wav2Vec2Config(vocab_size=len(tokenizer.txt2idx))
    
    