from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, Dataset
from jamo import h2j, j2hcj
from jamotools import join_jamos
import librosa
import numpy as np
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import SequentialSampler
import torch
from itertools import groupby
import re
import pandas as pd

def get_duration(file):
    #a,sr = librosa.load(file)
    #return len(a)
    return librosa.get_duration(filename=file)

class CustomTokenizer(object):
    def __init__(self, max_length=99999, max_vocab_size=-1):
        self.txt2idx = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }
        self.idx2txt = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.max_length = max_length
        self.char_count = {}
        self.max_vocab_size = max_vocab_size
        self.vocab = []

    def jamo_decode(self, sentence):
        jamo_sentence = j2hcj(h2j(sentence))
        return jamo_sentence

    def fit(self, sentence_list):
        for sentence in sentence_list:
            for char in self.jamo_decode(sentence):
                try:
                    self.char_count[char] += 1
                except:
                    self.char_count[char] = 1
        self.char_count = dict(
            sorted(self.char_count.items(), key=self.sort_target, reverse=True)
        )

        self.txt2idx = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }
        self.idx2txt = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        if self.max_vocab_size == -1:
            for i, char in enumerate(list(self.char_count.keys())):
                self.txt2idx[char] = i + 4
                self.idx2txt[i + 4] = char
        else:
            for i, char in enumerate(
                list(self.char_count.keys())[: self.max_vocab_size]
            ):
                self.txt2idx[char] = i + 4
                self.idx2txt[i + 4] = char

        self.vocab = list(self.txt2idx.keys())
        if len(self.vocab) < self.max_vocab_size:
            self.vocab.extend(['$'] * self.max_vocab_size - len(self.vocab))
            
    def sort_target(self, x):
        return x[1]

    def txt2token(self, sentence_list):
        tokens = []
        for j, sentence in enumerate(sentence_list):
            token = []
            token.append(self.txt2idx["<sos>"])
            for i, c in enumerate(self.jamo_decode(sentence)):
                # token = [0] * (self.max_length + 2)
                if i == self.max_length:
                    break
                try:
                    token.append(self.txt2idx[c])
                except:
                    token.append(self.txt2idx["<unk>"])
            try:
                token.append(self.txt2idx["<eos>"])
            except:
                pass
            tokens.append(token)

        return tokens

    def convert(self, tokens,predicted = True):
        if predicted:
            tokens = [token_group[0] for token_group in groupby(tokens)]
        #filtered_tokens = list(filter(lambda token: token != self.txt2idx["<pad>"] or token != self.txt2idx["<eos>"], tokens))

        sentence = []
        for i in tokens:
            if i == self.txt2idx["<pad>"]  or i == self.txt2idx["<sos>"] or i == -100:
                continue
            if i == self.txt2idx["<eos>"]:
                break
            try:
                sentence.append(self.idx2txt[i])
            except:
                sentence.append("<unk>")

        sentence = "".join(sentence).strip()
        # sentence = sentence[5:]

        return join_jamos(sentence)


class RandomBucketBatchSampler(object):
    """Yields of mini-batch of indices, sequential within the batch, random between batches.

    I.e. it works like bucket, but it also supports random between batches.
    Helpful for minimizing padding while retaining randomness with variable length inputs.
    Args:
        data_source (Dataset): dataset to sample from.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, data_source, batch_size, drop_last):
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integeral value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = SequentialSampler(
            data_source
        )  # impl sequential within the batch
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches()  # impl random between batches

    def _make_batches(self):
        indices = [i for i in self.sampler]
        # print(indices)
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        # print(batches)
        if self.drop_last and len(self.sampler) % self.batch_size > 0:
            random_indices = torch.randperm(len(batches) - 1).tolist() + [
                len(batches) - 1
            ]
        else:
            random_indices = torch.randperm(len(batches)).tolist()
        return [batches[i] for i in random_indices]

    def __iter__(self):
        for batch in self.random_batches:
            # print(batch)
            yield batch

    def __len__(self):
        return len(self.random_batches)


class with_kspon_datset(Dataset):
    def __init__(self, path_list, target_list=None,kspon_data = None, mode="train",sort = True, max_size = 0, min_size = 0):
        self.sr = 16000
        self.mode = mode
        self.metadata = pd.DataFrame()
        self.metadata['file'] = pd.Series(path_list)
        if self.mode == 'train':
            self.metadata['target'] = pd.Series(target_list)
            self.kspon_dataframe = kspon_data

        
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            do_normalize=True,
            feature_size=1,
            padding_side="right",
            padding_value=0.0,
            return_attention_mask=False,
            sampling_rate=16000,
        )

        if sort and self.mode == "train":
            print('getting duration of file')
            self.metadata['length'] = self.metadata['file'].apply(lambda x: get_duration(x))
            self.metadata = pd.concat([self.metadata,self.kspon_dataframe])
            self.metadata.sort_values(by=['length'], inplace=True, ascending=False)
            self.metadata = self.metadata[(self.metadata['length'] < max_size) & (self.metadata['length'] > min_size) ]

        if self.mode == 'train':
            print(self.metadata.head())
            print(self.metadata.tail())


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        data, rate = librosa.load(self.metadata.iloc[i]['file'], sr=None)
        if rate == 48000:
            data = librosa.resample(data, orig_sr=48000, target_sr=16000)
        """ sound = np.zeros(self.sound_max_length)
        if len(data) <= self.sound_max_length:
            sound[:data.shape[0]] = data
        else:
            sound = data[:self.sound_max_length]"""

        sound = self.feature_extractor(
            data, sampling_rate=16000, return_tensors="pt"
        ).input_values

        if self.mode == "train":
            text = self.metadata.iloc[i]['target']
            return torch.LongTensor(text), sound

        else:
            return sound

class CustomDataset(Dataset):
    def __init__(self, path_list, target_list=None, mode="train",sort = True, max_size = 0, min_size = 0):
        self.sr = 16000
        self.mode = mode
        self.metadata = pd.DataFrame()
        self.metadata['file'] = pd.Series(path_list)
        if self.mode == 'train':
            self.metadata['target'] = pd.Series(target_list)

        self.feature_extractor = Wav2Vec2FeatureExtractor(
            do_normalize=True,
            feature_size=1,
            padding_side="right",
            padding_value=0.0,
            return_attention_mask=False,
            sampling_rate=16000,
        )

        if sort and self.mode == "train":
            print('getting duration of file')
            self.metadata['length'] = self.metadata['file'].apply(lambda x: get_duration(x))
            self.metadata.sort_values(by=['length'], inplace=True, ascending=False)
            self.metadata = self.metadata[(self.metadata['length'] < max_size) & (self.metadata['length'] > min_size) ]

        if self.mode == 'train':
            print(self.metadata.head())
            print(self.metadata.tail())


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        data, rate = librosa.load(self.metadata.iloc[i]['file'], sr=None)
        data = librosa.resample(data, orig_sr=48000, target_sr=16000)
        """ sound = np.zeros(self.sound_max_length)
        if len(data) <= self.sound_max_length:
            sound[:data.shape[0]] = data
        else:
            sound = data[:self.sound_max_length]"""

        sound = self.feature_extractor(
            data, sampling_rate=16000, return_tensors="pt"
        ).input_values

        if self.mode == "train":
            text = self.metadata.iloc[i]['target']
            return torch.LongTensor(text), sound

        else:
            return sound

class AudioCollate(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        # Right zero-pad mel-spec
        max_target_len = max([x.size(1) for x in batch])
        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), max_target_len)
        mel_padded.zero_()
        # output_lengths = torch.LongTensor(len(batch))

        
        for i in range(len(batch)):
            mel = batch[i]
            mel_padded[i, : mel.size(1)] = mel
            # mel_padded[i,mel.size(1):] = -100
            # output_lengths[i] = mel.size(1)

        # mask = get_mask_from_lengths(output_lengths)
        
        return {"speech": mel_padded}


class TextAudioCollate(object):
    """Another way to implement collate_fn passed to DataLoader.
    Use class but not function because this is easier to pass some parameters.
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Process one mini-batch samples, such as sorting and padding.
        Args:
            batch: a list of (text sequence, audio feature sequence, filename)
        Returns:
            text_padded: [N, Ti]
            input_lengths: [N]
            mel_padded: [N, To, D]
            gate_padded: [N, To]
            output_lengths: [N]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        # print(input_lengths)

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text
            text_padded[
                i, text.size(0) :
            ] = -100  # https://huggingface.co/transformers/model_doc/wav2vec2.html?highlight=wav2vec2forctc#wav2vec2forctc

        # Right zero-pad mel-spec
        # num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), max_target_len)
        mel_padded.zero_()
        # output_lengths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, : mel.size(1)] = mel
            # mel_padded[i,mel.size(1):] = -100
            # output_lengths[i] = mel.size(1)

        # mask = get_mask_from_lengths(output_lengths)

        return {"speech": mel_padded, "labels": text_padded}


def get_mask_from_lengths(lengths):
    """Mask position is set to 1 for Tensor.masked_fill(mask, value)"""
    N = lengths.size(0)
    T = torch.max(lengths).item()
    mask = torch.zeros(N, T)
    for i in range(N):
        mask[i, lengths[i] :] = 1
    return mask.bool()


if __name__ == "__main__":

    path_list = ["../hobby_00027892.wav", "../hobby_00031952.wav"]
    target_list = ["안녕하세요", "저는 이도운입니다."]
    tokenizer = CustomTokenizer()
    tokenizer.fit(target_list)
    token = tokenizer.txt2token(target_list)
    print("*" * 80)
    print(token)
    dataset = CustomDataset(path_list, token)
    batch_sampler = RandomBucketBatchSampler(dataset, batch_size=5, drop_last=False)
    collate_fn = TextAudioCollate()
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    print(tokenizer.idx2txt)
    """for i, data in enumerate(dataset):
        audio,text = data
        #print(text)
        #print(audio)
        # print(i, len(text), np.array(audio).size())
        print(type(audio))
        print(np.array(audio).shape)"""

    for i, batch in enumerate(dataloader):
        """print(batch[0][0].shape)
        print(batch[0][1].shape)
        print(batch[1][0].shape)
        print(batch[1][1].shape)
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[1])"""
        print(batch["speech"][1])
        print(batch["speech"][1].shape)
