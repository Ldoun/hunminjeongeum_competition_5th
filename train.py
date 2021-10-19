#!/usr/bin/env python3
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
from ctcdecode import CTCBeamDecoder

def evaluate(model, batch, device):
    model.to(device)
    # as the target is english, the first word to the transformer should be the
    # english start token.
    tokenizer = dict_for_infer["tokenizer"]

    alpha=0
    beta=0
    beam_width = 100

    beam_decoder = CTCBeamDecoder(tokenizer.vocab ,
                                 alpha=alpha, beta=beta,
                                 cutoff_top_n=40, cutoff_prob=1.0,
                                 beam_width=beam_width, num_processes=7,
                                 blank_id=tokenizer.txt2idx["<pad>"])

    
    with torch.no_grad():
        logits = model(batch).logits

    beam_results, beam_scores, timesteps, out_lens = beam_decoder.decode(logits)

    result_list = []
    for token in beam_results:
        result_list.append("".join([tokenizer.idx2txt[x] for x in token[0]]))

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

        model.load_state_dict(checkpoint["model"])

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), "rb") as f:
            dict_for_infer = pickle.load(f)

        print("로딩 완료!")

    def infer(test_path, **kwparser):
        device = checkpoint["device"]
        test_file_list = sorted(
            glob(os.path.join(DATASET_PATH, "test", "test_data", "*"))
        )

        test_dataset = CustomDataset(path_list=test_file_list, mode="test")
        test_sampler = RandomBucketBatchSampler(
            test_dataset, batch_size=dict_for_infer["batch_size"], drop_last=False
        )
        callate_fn = AudioCollate()
        test_data_loader = DataLoader(
            test_dataset, batch_sampler=test_sampler, collate_fn=callate_fn, num_workers=7,pin_memory=True
        )

        result_list = []

        for step, batch in enumerate(test_data_loader):
            speech = batch["speech"].to(device)
            output = evaluate(model, speech, device)
            result_list.extend(output)

        prob = [1] * len(result_list)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        # return list(zip(pred.flatten(), clipped.flatten()))
        return list(zip(prob, result_list))

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
                                 beam_width=beam_width, num_processes=7,
                                 blank_id=tokenizer.txt2idx["<pad>"])


    for i, batch in enumerate(tqdm(valid_dataloader)):
        print("validation:" + str(i) + "/" + str(total))
        with torch.no_grad():
            speech = batch["speech"].to(device)
            text = batch["labels"].to(device)

            model_predictions = model(speech, labels=text).logits

            print(model_predictions.shape)
            #predicted_ids = torch.argmax(model_predictions, dim=-1)
            
            #predictions = [
            #    tokenizer.convert(sen) for sen in predicted_ids.cpu().numpy()
            #]
        beam_results, beam_scores, timesteps, out_lens = beam_decoder.decode(model_predictions)
        result_list = []
        
        for token in beam_results:
            result_list.append("".join([tokenizer.idx2txt[x] for x in token[0]]))

        references = [tokenizer.convert(sen,predicted=False) for sen in text.cpu().numpy()]
        metric.add_batch(predictions=result_list, references=references)

    final_score = metric.compute()

    return {"cer": final_score}


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
    parser.add_argument("--warmup_step", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--reload_from", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--valid_every", type=int, default=700)
    parser.add_argument("--save_every", type=int, default=700)
    parser.add_argument("--strategy", type=str, default="step")
    parser.add_argument("--max_vocab_size", type=int, default=80)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--session", type=str)

    args = parser.parse_args()

    global dict_for_infer

    config = Wav2Vec2Config(vocab_size=args.max_vocab_size)
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
    model.freeze_feature_extractor()
    model.config = config
    model.lm_head = nn.Linear(
        in_features=768, out_features=args.max_vocab_size, bias=True
    )

    bind_model(model=model, parser=args)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == "train":
       
        train_path = os.path.join(DATASET_PATH, "train")
        file_list = sorted(glob(os.path.join(train_path, "train_data", "*")))
        label = pd.read_csv(os.path.join(train_path, "train_label"))

        split_num = int(len(label) * 0.9)
        train_file_list = file_list[:split_num]
        val_file_list = file_list[split_num:]

        train_label = label.iloc[:split_num]
        val_label = label.iloc[split_num:]


        if args.reload_from != 0:
            nsml.load(args.checkpoint, session = args.session)
            tokenizer = dict_for_infer['tokenizer']

            args.batch_size = dict_for_infer['batch_size']
            args.total_epoch = dict_for_infer['epochs']
            args.lr = dict_for_infer['learning_rate']


        else:
            tokenizer = CustomTokenizer()
            tokenizer.fit(train_label.text)

        train_tokens = tokenizer.txt2token(train_label.text)
        valid_tokens = tokenizer.txt2token(val_label.text)

        train_dataset = CustomDataset(train_file_list, train_tokens)
        valid_dataset = CustomDataset(val_file_list, valid_tokens)

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
            num_workers=7,
            pin_memory=True
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_sampler=valid_batch_sampler,
            collate_fn=collate_fn,
            num_workers=7,
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
