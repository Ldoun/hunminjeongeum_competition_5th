#!/usr/bin/env python3
from re import A
from shutil import ExecError
import numpy as np
import torch
import torch.nn as nn
import time
from packaging import version
import math
from dataloader_np import *
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
import json
from pydub import AudioSegment
from pydub.silence import split_on_silence

from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from apex import amp
from ctcdecode import CTCBeamDecoder

def evaluate(model, batch,tokenizer, beam_decoder):
    model.eval()
    
    with torch.no_grad():
        logits = model(batch).logits

    beam_results, beam_scores, timesteps, out_lens = beam_decoder.decode(logits)

    result_list = []
    for token,out_len in zip(beam_results.cpu().numpy(),out_lens):
        a = tokenizer.convert(token[0][:out_len[0]],predicted=False)
        result_list.append(a)

    return result_list

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
        
        files = []
        
        for file_num, file in enumerate(test_file_list):
            addition_flag = []
            result_chunk = []
            sound = AudioSegment.from_wav(file)
            sound = match_target_amplitude(sound, -20.0)
            
            split_to_chunk(sound,result_chunk,addition_flag,500)
            
            result = []
            for i, flag in enumerate(addition_flag):
                if flag == 1 and i < len(addition_flag) -1:
                    result_chunk[i+1] = result_chunk[i] + result_chunk[i + 1]
                elif flag == 0 or result_chunk[i].frame_count() > 80000:
                    result.append(result_chunk[i])
                elif i == len(addition_flag) -1:
                    result[-1] = result[-1] + result_chunk[i]

            for j, chunk in enumerate(result):
                new_file = str(file_num) + '_' + str(j)
                np.save(new_file, chunk.get_array_of_samples())
                files.append(new_file + '.npy')
        
        test_data = pd.DataFrame()
        test_data['file'] = pd.Series(files)
        
        test_dataset = CustomDataset(test_data, mode="test")
        test_sampler = RandomBucketBatchSampler(
            test_dataset, batch_size=dict_for_infer["batch_size"], drop_last=False
        )
        callate_fn = AudioCollate()
        test_data_loader = DataLoader(
            test_dataset,batch_size=dict_for_infer["batch_size"], collate_fn=callate_fn, num_workers=8,pin_memory=True
        )

        tokenizer  = dict_for_infer["tokenizer"]
        beam_decoder = CTCBeamDecoder(tokenizer.vocab,
                                #model_path='n-gram/stt2_n2.binary',
                                alpha=0, beta=0,
                                cutoff_top_n=20, cutoff_prob=1.0,
                                beam_width=100, num_processes=4,
                                blank_id=tokenizer.txt2idx["<pad>"],
                                log_probs_input=True)
        
        model.to(device)
        if args.fp16 and args.mode == 'test':
            model2 = amp.initialize(model, opt_level="O1")
            print('fp16 on')
        else:
            model2 = model
        model2.load_state_dict(checkpoint["model"])
        
        result_list = []
        result_file_list = []
        
        for step, batch in enumerate(test_data_loader):
            speech = batch["speech"].to(device)
            output = evaluate(model2, speech,tokenizer, beam_decoder)
            
            result_list.extend(output)
            result_file_list.extend(batch['file'])
            
        final_result = [str() for i in range(len(test_file_list))]
        for r,f in zip(result_list,result_file_list):
            num = int(f.split('_')[0])
            final_result[num] = final_result[num] + r + ' '

        for i in range(len(final_result)):
            final_result[i] = final_result[i].strip()
        
        files = []
        for file in test_file_list:
            files.append(file.split('/')[-1])

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        # return list(zip(pred.flatten(), clipped.flatten()))
        return list(zip(files, final_result))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def validate(valid_dataloader, model, tokenizer):
    model.eval()
    device = next(model.parameters()).device
    metric = load_metric("cer")

    total = len(valid_dataloader)

    alpha=0
    beta=0
    beam_width = 100
    
    beam_decoder = CTCBeamDecoder(tokenizer.vocab,
                                 alpha=alpha, beta=beta,
                                 cutoff_top_n=40, cutoff_prob=1.0,
                                 beam_width=beam_width, num_processes=11,
                                 blank_id=tokenizer.txt2idx["<pad>"],
                                 log_probs_input=True)


    for i, batch in enumerate(tqdm(valid_dataloader)):
        print("validation:" + str(i) + "/" + str(total))
        with torch.no_grad():
            speech = batch["speech"].to(device)
            text = batch["labels"].to(device)
            
            print(speech.shape)

            model_predictions = model(speech, labels=text).logits

        '''predicted_ids = torch.argmax(model_predictions, dim=-1)

        
        predictions = [
            tokenizer.convert(sen) for sen in predicted_ids.cpu().numpy()
        ]'''

        beam_results, beam_scores, timesteps, out_lens = beam_decoder.decode(model_predictions)
        result_list = []
        
        for token,out_len in zip(beam_results.cpu().numpy(),out_lens):
            #result_list.append("".join([tokenizer.idx2txt[x] for x in token]))
            a = tokenizer.convert(token[0][:out_len[0]],predicted=False)
            result_list.append(a)

        references = [tokenizer.convert(sen,predicted=False) for sen in text.cpu().numpy()]
        
        print(result_list[0])

        metric.add_batch(predictions=result_list, references=references)
        
    final_score = metric.compute()
    print(final_score)

    return {"cer": final_score}

def clean(sen):
    cleaned_sen = re.sub('SP|FP|SN|NO|\(|\)|:|\*|,|…|\{[^\}]+\}','',sen)
    cleaned_sen = re.sub('&[^&]+&','m',cleaned_sen)
    cleaned_sen = re.sub('\s{2,}',' ',cleaned_sen)
    return cleaned_sen

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default="0")
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--total_epoch", type=int, default=100)
    parser.add_argument("--warmup_step", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=5e-5)
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

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
    model.freeze_feature_extractor()

    bind_model(model=model, parser=args)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == "train":
       
        train_path = os.path.join(DATASET_PATH, "train")
        wav_path = os.path.join(train_path, "train_data","wav")
        json_list = sorted(glob(os.path.join(train_path, "train_data","train_info", "*")))
        label = pd.read_csv(os.path.join(train_path, "train_label"))

        split_num = int(len(label) * 0.93)
        
        train_label = label.iloc[:split_num]
        val_label = label.iloc[split_num:]
        
        
        splited_train_labels = []
        splited_valid_labels = []
        splited_train_file = []
        splited_valid_file = []
        train_duration = []
        valid_duration = []
        file_cnt = 0
        
        for file in json_list:
            with open(file,'r',encoding='UTF8') as f:
                data = json.load(f)
                
            #sound, sr = librosa.load(os.path.join(wav_path,data['id']), sr=None)
            sound = AudioSegment.from_wav(os.path.join(wav_path,data['id']))
            sound = match_target_amplitude(sound, -20.0).get_array_of_samples()
            sr = 16000
            
            for value in data['utterance']:
                np_sound = sound[int(value['start'] * sr): int(value['end'] * sr)]
                
                if data['id'] in train_label['file_name'].values:
                    splited_train_labels.append(value['dialect_form'])
                    file_name = str(file_cnt) 
                    splited_train_file.append(file_name + '.npy')
                    train_duration.append(len(np_sound)) 
                    
                elif data['id'] in val_label['file_name'].values:
                    splited_valid_labels.append(value['dialect_form'])
                    file_name = str(file_cnt) 
                    splited_valid_file.append(file_name + '.npy')
                    valid_duration.append(len(np_sound)) 
                    
                file_cnt += 1
                
                np.save(file_name,np_sound)
        
                
        train_data = pd.DataFrame()
        train_data['file'] = pd.Series(splited_train_file)
        train_data['target'] = pd.Series(splited_train_labels)
        train_data['length'] = pd.Series(train_duration)
       
        valid_data = pd.DataFrame()
        valid_data['file'] = pd.Series(splited_valid_file)
        valid_data['target'] = pd.Series(splited_valid_labels)  
        valid_data['length'] = pd.Series(valid_duration)  
        
        train_label = [clean(sen) for sen in train_data.target]
        val_label = [clean(sen) for sen in valid_data.target]

        if args.reload_from != 0:
            nsml.load(args.checkpoint, session = args.session)
            tokenizer = dict_for_infer['tokenizer']

            args.batch_size = dict_for_infer['batch_size']
            args.total_epoch = dict_for_infer['epochs']
            args.lr = dict_for_infer['learning_rate']

        else:
            tokenizer = CustomTokenizer()
            tokenizer.fit(train_label)

        print(tokenizer.txt2idx)

        model.lm_head = nn.Linear(
            in_features=768, out_features=len(tokenizer.txt2idx), bias=True
        )
        model.config = Wav2Vec2Config(vocab_size=len(tokenizer.txt2idx))

        train_tokens = tokenizer.txt2token(train_label)
        valid_tokens = tokenizer.txt2token(val_label)
        
        train_data['target'] = pd.Series(train_tokens)
        valid_data['target'] = pd.Series(valid_tokens)

        train_dataset = CustomDataset(train_data, max_size=200000, min_size=5000)
        valid_dataset = CustomDataset(valid_data, max_size=200000, min_size=5000)        
        
        print(len(train_dataset))
        print('-'*80)

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
            num_workers=11,
            pin_memory=True
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_sampler=valid_batch_sampler,
            collate_fn=collate_fn,
            num_workers=11,
            pin_memory=True
        )

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        ###############################################################################
        # Prepare the Optimizer
        ###############################################################################

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_step,
            num_training_steps=len(train_dataloader) * args.total_epoch,
        )  # do not foget to modify the number when dataset is changed

        model.to(device)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )   
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        
        if args.reload_from != 0:
            optimizer.load_state_dict(dict_for_infer['opt'])
            model.load_state_dict(dict_for_infer['model'])
            scheduler.load_state_dict(dict_for_infer['scaler'])
            if args.fp16:
                amp.load_state_dict(checkpoint['amp'])

        n_iters = len(train_dataloader)

        if args.strategy == "epoch":
            args.valid_every = n_iters
            args.save_every = n_iters

        itr_global = args.reload_from + 1
        for epoch in range(int(args.reload_from / n_iters) + 1, args.total_epoch + 1):
            itr_start_time = time.time()
            losses = []
            for batch in train_dataloader:
                model.train()
                speech = batch["speech"].to(device)
                text = batch["labels"].to(device)

                loss = model(speech, labels=text).loss

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                losses.append(loss.item())

                if itr_global % args.log_every == 0:
                    elapsed = time.time() - itr_start_time
                    print(
                        "epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f"
                        % ( 
                            epoch,
                            args.total_epoch,
                            itr_global % n_iters,
                            n_iters,
                            elapsed,
                            np.mean(losses),
                        )
                    )

                    summary = {"summary": True, "scope": locals(), "step": itr_global}
                    summary.update({"loss": np.mean(losses)})
                    nsml.report(**summary)

                    losses = []
                    itr_start_time = time.time()

                itr_global = itr_global + 1

                if itr_global % args.valid_every == 0:
                    print("validating..")
                    valid_result = validate(valid_dataloader, model, tokenizer)
                    print(valid_result)

                    summary = {"summary": True, "scope": locals(), "step": itr_global}
                    summary.update(valid_result)
                    nsml.report(**summary)

                dict_for_infer = {
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "scaler": scheduler.state_dict(),
                    "amp": amp.state_dict(),
                    "batch_size": args.batch_size,
                    "epochs": args.total_epoch,
                    "learning_rate": args.lr,
                    "tokenizer": tokenizer,
                    "device": device,
                }

                if itr_global % args.save_every == 0:
                    print("saving...")
                    nsml.save(checkpoint=f"{itr_global}")
