import json
import pandas as pd
from glob import iglob,glob
import os
from tqdm import tqdm
from multiprocessing import freeze_support, Pool
import re

def read(file):
    with open(file, 'r', encoding='UTF8') as reader:
        json_data = json.load(reader)
    
    return json_data

def clean(sen):
    cleaned_sen = re.sub('SP|FP|SN|NO|\(|\)|:|\*|,|…|\{[^\}]+\}','',sen)
    cleaned_sen = re.sub('&[^&]+&','m',cleaned_sen)
    cleaned_sen = re.sub('\s{2,}',' ',cleaned_sen)
    return cleaned_sen

if __name__ == '__main__':
    path = '/home/Lee/ai-hub_competition/ngram/stt_3/json_file' 
    json_list = glob(os.path.join(path, "*.json"))
    
    result = []
    for file in tqdm(json_list):
        sentence = []
        for value in file['utterance']:
            sentence.append(value['dialect_form'])
            
        result.append(clean(' '.join(sentence)) + '\n')
        
    with open('target.txt','w','UTF8') as writer:
        writer.write(result)
        
        