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

def evaluate(model, batch,tokenizer, beam_decoder):
    model.eval()
    with torch.no_grad():
        logits = model(batch).logits

    beam_results, beam_scores, timesteps, out_lens = beam_decoder.decode(logits)
    
    '''print(len(beam_results[0]))
    print(len(beam_results))
    print(beam_results.shape)
    print('-'*80)
    print(beam_scores)

    print(list(beam_results[0][0][:out_lens[0][0]].cpu().numpy()))'''
    
    result_list = []
    for token,out_len in zip(beam_results.cpu().numpy(),out_lens):
        a = tokenizer.convert(token[0][:out_len[0]],predicted=False)
        result_list.append(a)

    return result_list


def save_checkpoint(checkpoint, dir):
    torch.save(checkpoint, os.path.join(dir))


def bind_model(model, parser):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *parser):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, "checkpoint")
        
        save_checkpoint(dict_for_infer, save_dir)

        with open(os.path.join(dir_name, "dict_for_infer"), "wb") as f:
            pickle.dump(dict_for_infer, f)

        print("저장 완료!")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *parser):

        save_dir = os.path.join(dir_name, "checkpoint")

        global checkpoint
        checkpoint = torch.load(save_dir)

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), "rb") as f:
            dict_for_infer = pickle.load(f)

        tokenizer = dict_for_infer["tokenizer"]
        model.lm_head = nn.Linear(
            in_features=768, out_features=len(tokenizer.txt2idx), bias=True
        )
        model.config = Wav2Vec2Config(vocab_size=len(tokenizer.txt2idx))
        
        #model.load_state_dict(checkpoint["model"])

        print("로딩 완료!")

    def infer(test_path, **kwparser):
        device = checkpoint["device"]
        test_file_list = sorted(
            glob(os.path.join(DATASET_PATH, "test", "test_data", "*"))
        )

        test_dataset = CustomDataset(test_file_list, mode="test")
        test_sampler = RandomBucketBatchSampler(
            test_dataset, batch_size=dict_for_infer["batch_size"], drop_last=False
        )
        callate_fn = AudioCollate()
        test_data_loader = DataLoader(
            test_dataset,batch_size=dict_for_infer["batch_size"], collate_fn=callate_fn, num_workers=8,pin_memory=True
        )
        
        alpha=0.5
        beta=0.3
        beam_width = 30
        
        tokenizer  = dict_for_infer["tokenizer"]

        beam_decoder = CTCBeamDecoder(tokenizer.vocab,
                                #model_path='n-gram/stt2_n2.binary',
                                alpha=0, beta=0,
                                cutoff_top_n=10, cutoff_prob=1.0,
                                beam_width=beam_width, num_processes=4,
                                blank_id=tokenizer.txt2idx["<pad>"],
                                log_probs_input=True)

        result_list = []
        
        model.to(device)
        if args.fp16 and args.mode == 'test':
            model2 = amp.initialize(model, opt_level="O1")
            print('fp16 on')
        else:
            model2 = model

        model2.load_state_dict(checkpoint["model"])
        
        for step, batch in enumerate(test_data_loader):
            speech = batch["speech"].to(device)
            output = evaluate(model2, speech,tokenizer, beam_decoder)
            result_list.extend(output)

        prob = [1] * len(result_list)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        # return list(zip(pred.flatten(), clipped.flatten()))
        return list(zip(prob, result_list))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

def clean(sen):
    cleaned_sen = re.sub('SP|FP|SN|NO|\(|\)|:|\*|,|…','',sen)
    cleaned_sen = re.sub('\s{2,}',' ',cleaned_sen)
    return cleaned_sen

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def split_to_chunk(speech,result_chunk,addition_flag,min_silence_len):
    if min_silence_len < 200:
        return
    audio_chunks = split_on_silence(speech, min_silence_len=min_silence_len, silence_thresh=-40)
    for i,chunk in enumerate(audio_chunks):
        if chunk.frame_count() > 200000:
            split_to_chunk(chunk,result_chunk, addition_flag, min_silence_len-100)

        else:
            result_chunk.append(chunk)
            if chunk.frame_count() < 16000:
                addition_flag.append(1)
            else:
                addition_flag.append(0)
    return


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default="0")
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--total_epoch", type=int, default=40)
    parser.add_argument("--warmup_step", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--reload_from", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--valid_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--strategy", type=str, default="step")
    parser.add_argument("--max_vocab_size", type=int, default=-1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--session", type=str)

    args = parser.parse_args()

    global dict_for_infer

    model = None

    bind_model(model=model, parser=args)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == "train":
        '''print(os.listdir('./'))
        kspon_data = pd.read_csv('kspon_data.tsv',sep='\t')
        duration = sorted([get_duration(file) for file in kspon_data['file']])
        
        print('start measuring duration')
        print('0%:', str(duration[:10]))
        print('50%: ',str(duration[int(len(duration) * 0.5)]))
        print('80%: ',str(duration[int(len(duration) * 0.8)]))
        print('90%: ',str(duration[int(len(duration) * 0.9)]))
        print('95%: ',str(duration[int(len(duration) * 0.95)]))
        print('98%: ',str(duration[int(len(duration) * 0.98) -1]))
        print('99%: ',str(duration[int(len(duration) * 0.99) -1]))
        print('99.5%: ',str(duration[int(len(duration) * 0.995) -1]))
        print('100%: ',str(duration[int(len(duration)) -1]))
        
        print(len(duration) - int(len(duration) * 0.995))
        print(len(duration))
        
        stt2_kspon = [length for length in duration if length < 7.62]
        stt1_kspon = [length for length in duration if length * 16000 < 213347]
        
        print('stt1','*'*40 )
        print('start measuring duration')
        print('0%:', str(stt1_kspon[:10]))
        print('50%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.5)]))
        print('80%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.8)]))
        print('90%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.9)]))
        print('95%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.95)]))
        print('98%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.98) -1]))
        print('99%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.99) -1]))
        print('99.5%: ',str(stt1_kspon[int(len(stt1_kspon) * 0.995) -1]))
        print('100%: ',str(stt1_kspon[int(len(stt1_kspon)) -1]))
        
        print(len(stt1_kspon) - int(len(stt1_kspon) * 0.995))
        print(len(stt1_kspon))
        
        print('stt2','*'*40 )
        print('start measuring duration')
        print('0%:', str(stt2_kspon[:10]))
        print('50%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.5)]))
        print('80%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.8)]))
        print('90%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.9)]))
        print('95%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.95)]))
        print('98%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.98) -1]))
        print('99%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.99) -1]))
        print('99.5%: ',str(stt2_kspon[int(len(stt2_kspon) * 0.995) -1]))
        print('100%: ',str(stt2_kspon[int(len(stt2_kspon)) -1]))
        
        print(len(stt2_kspon) - int(len(stt2_kspon) * 0.995))
        print(len(stt2_kspon))
        '''
        print(os.listdir())
        raise       
        train_path = os.path.join(DATASET_PATH, "train")
        file_list = sorted(glob(os.path.join(train_path, "train_data","wav", "*")))
        label = pd.read_csv(os.path.join(train_path, "train_label"))
        json_list = sorted(glob(os.path.join(train_path, "train_data", "train_info", "*")))
        
        files = []
        for file_num, file in enumerate(file_list):
            addition_flag = []
            result_chunk = []
            sound = AudioSegment.from_wav(file)
            sound = match_target_amplitude(sound, -20.0)
            
            split_to_chunk(sound,result_chunk,addition_flag,500)
            
            print(sound.frame_count())
            print(str(len(addition_flag)) +' ' + str(len(result_chunk)))
            print('-'*80)
            
            result = []
            for i, flag in enumerate(addition_flag):
                if flag == 1 and i < len(addition_flag) -1:
                    result_chunk[i+1] = result_chunk[i] + result_chunk[i + 1]
                elif flag == 0 or result_chunk[i].frame_count() > 32000:
                    result.append(result_chunk[i])
                elif i == len(addition_flag) -1:
                    result[-1] = result[-1] + result_chunk[i]

            for j, chunk in enumerate(result):
                new_file = str(file_num) + '_' + str(j)
                np.save(new_file, chunk.get_array_of_samples())
                files.append(new_file + '.npy')
                
                
        
        labels = []
        for file in json_list:
            with open(file,'r',encoding='UTF8') as f:
                data = json.load(f)
                
            for value in data['utterance']:
                labels.append(value['dialect_form'])
                
        labels = [char for sen in labels for char in sen ]
        print(set(labels))
        print(len(set(labels))) 
           
        counter = {}
        for sen in labels:
            for char in sen:
                try:
                    counter[char] += 1
                except:
                    counter[char] = 0
                    
        length = sorted(list(counter.values()))
        
        print('0%:', str(length[:10]))
        print('50%: ',str(length[int(len(length) * 0.5)]))
        print('80%: ',str(length[int(len(length) * 0.8)]))
        print('90%: ',str(length[int(len(length) * 0.9)]))
        print('95%: ',str(length[int(len(length) * 0.95)]))
        print('98%: ',str(length[int(len(length) * 0.98) -1]))
        print('99%: ',str(length[int(len(length) * 0.99) -1]))
        print('99.5%: ',str(length[int(len(length) * 0.995) -1]))
        print('100%: ',str(length[int(len(length)) -1]))
        
        
        print(len([x for x in length if x == 0]))
        raise
        

        split_num = int(len(label) * 0.9)
        train_file_list = file_list[:split_num]
        val_file_list = file_list[split_num:]

        train_label = label.iloc[:split_num]
        val_label = label.iloc[split_num:]
                

        train_label = [clean(sen) for sen in train_label.text]
        val_label = [clean(sen) for sen in val_label.text]
        labels = [clean(sen) for sen in labels]

        if args.reload_from != 0:
            nsml.load(args.checkpoint, session = args.session)
            tokenizer = dict_for_infer['tokenizer']

            args.batch_size = dict_for_infer['batch_size']
            args.lr = dict_for_infer['learning_rate']

        else:
            tokenizer = CustomTokenizer()
            tokenizer.fit(train_label)

        print(tokenizer.txt2idx)

        train_tokens = tokenizer.txt2token(train_label)
        valid_tokens = tokenizer.txt2token(val_label)
        total_tokens = tokenizer.txt2token(labels)
        
        length = sorted([len(token) for token in total_tokens])
        
        print('start measuring duration')
        print('0%:', str(length[:10]))
        print('50%: ',str(length[int(len(length) * 0.5)]))
        print('80%: ',str(length[int(len(length) * 0.8)]))
        print('90%: ',str(length[int(len(length) * 0.9)]))
        print('95%: ',str(length[int(len(length) * 0.95)]))
        print('98%: ',str(length[int(len(length) * 0.98) -1]))
        print('99%: ',str(length[int(len(length) * 0.99) -1]))
        print('99.5%: ',str(length[int(len(length) * 0.995) -1]))
        print('100%: ',str(length[int(len(length)) -1]))
        
        print(len(length) - int(len(length) * 0.995))
        print(len(length))
                
                
        raise
        model.lm_head = nn.Linear(
            in_features=768, out_features=len(tokenizer.txt2idx), bias=True
        )
        model.config = Wav2Vec2Config(vocab_size=len(tokenizer.txt2idx))


        train_dataset = CustomDataset(train_file_list, train_tokens, max_size=7.62, min_size=0)
        valid_dataset = CustomDataset(val_file_list, valid_tokens,  max_size=7.62, min_size=0)

        train_batch_sampler = RandomBucketBatchSampler(
            train_dataset, batch_size=args.batch_size, drop_last=False
        )
        valid_batch_sampler = RandomBucketBatchSampler(
            valid_dataset, batch_size=args.batch_size, drop_last=False
        )

        collate_fn = TextAudioCollate()

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_sampler=valid_batch_sampler,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True
        )

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        
        model.to(device)
        model = amp.initialize(model, opt_level="O1")
        
        if args.reload_from != 0:
            model.load_state_dict(dict_for_infer['model'])

        

        device = checkpoint["device"]
        test_file_list = sorted(
            glob(os.path.join(train_path, "train_data", "*"))
        )

        test_dataset = CustomDataset(test_file_list, mode="test")
        
        callate_fn = AudioCollate()
        test_data_loader = DataLoader(
            test_dataset,batch_size=dict_for_infer["batch_size"], collate_fn=callate_fn, num_workers=8,pin_memory=True
        )
        
        alpha=0.2
        beta=0.4
        beam_width = 30
        
        tokenizer  = dict_for_infer["tokenizer"]
        print('With LM')

        LM_beam_decoder = CTCBeamDecoder(tokenizer.vocab,
                                model_path='n-gram/stt2_n2.binary',
                                alpha=alpha, beta=beta,
                                cutoff_top_n=10, cutoff_prob=1.0,
                                beam_width=beam_width, num_processes=4,
                                blank_id=tokenizer.txt2idx["<pad>"],
                                log_probs_input=True)
        
        '''noLM_beam_decoder = CTCBeamDecoder(tokenizer.vocab,
                                #model_path='n-gram/stt2_n2.binary',
                                alpha=alpha, beta=beta,
                                cutoff_top_n=10, cutoff_prob=1.0,
                                beam_width=beam_width, num_processes=4,
                                blank_id=tokenizer.txt2idx["<pad>"],
                                log_probs_input=True)'''

        result_list = []
        
        model.to(device)
        if args.fp16 and args.mode == 'test':
            model2 = amp.initialize(model, opt_level="O1")
            print('fp16 on')
        else:
            model2 = model

        model2.load_state_dict(checkpoint["model"])
        
        
        now = time.time()
        time_list = []
        for step, batch in enumerate(test_data_loader):
            print(step)
            if step>100:
                break
            speech = batch["speech"].to(device)
            
            model.eval()
            with torch.no_grad():
                logits = model(speech).logits

            #predicted_ids = torch.argmax(logits, dim=-1)
            #output = predicted_ids.cpu().numpy()
            #print(output)
            
            now_lm = time.time()
            beam_results, beam_scores, timesteps, out_lens = LM_beam_decoder.decode(logits)
            print('LM:', str(time.time() - now_lm))
            
            '''now_nolm = time.time()
            beam_results, beam_scores, timesteps, out_lens = noLM_beam_decoder.decode(logits)
            print('noLM:', str(time.time() - now_nolm))
            print('-'*80)'''
                        
            result_list = []
            for token,out_len in zip(beam_results.cpu().numpy(),out_lens):
                a = tokenizer.convert(token[0][:out_len[0]],predicted=False)
                result_list.append(a)
                
            #print(result_list)
            #result_list.extend(output)
            #print(result_list)
        
        
        #print('final:', time.time() - now)
        #print('ave', sum(time_list) / len(time_list))