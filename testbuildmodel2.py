from flask import Flask
import pandas as pd
import numpy as np
from glob import glob
from ast import literal_eval
from tqdm import tqdm_notebook
from collections import Counter
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from pythainlp.ulmfit import *
from pythainlp.tokenize import Tokenizer
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback

#from pythainlp.ulmfit.tokenizer import ThaiTokenizer
import csv
####
#stock_files = sorted(glob("testmulticsv/*.csv"))
#stock_files
#print(stock_files)
###
#all_df = pd.concat((pd.read_csv(file)
 #         for file in stock_files), ignore_index = True)

#print(all_df)
if __name__ == '__main__':
    print("Start import json")
####
###    from sklearn.model_selection import train_test_split
###    all_df = pd.read_json("https://api.kaojao.com/api/facebook/facebookcomment/list?page=0")
###    train_df, valid_df =  train_test_split(all_df,test_size=0.15, random_state=1452)
####
##    print("Change type to string")
#change type int to string
 #   all_df['type'] = all_df['type'].replace([1],'pos')
#    all_df['type'] = all_df['type'].replace([2],'neg')
#    all_df['type'] = all_df['type'].replace([3],'neu')
 #   all_df['type'] = all_df['type'].replace([4],'qes')
    from sklearn.model_selection import train_test_split
    all_df = pd.read_csv("/home/az2-user/model/allreadjson.csv")
    train_df, valid_df =  train_test_split(all_df,test_size=0.15, random_state=1452)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
#tokenizer data
    print("Tokenizer data")
    tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
    processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000,mark_fields=False),
             NumericalizeProcessor(vocab=None, max_vocab=60000, min_freq=2)]

#path model

    model_path = '/home/az2-user/model' 
#build model
    print("start build model")
    data_lm = (TextList.from_df(all_df, model_path, cols="text", processor=processor)
        .split_by_rand_pct( valid_pct = 0.1 , seed = 1412)#valid_pct = 0.01,
        .label_for_lm()
        .databunch(bs=48))
    print("before save build model")
    data_lm.sanity_check()
    data_lm.save('/home/az2-user/model/model_pkl.pkl')
    print("after save build")
##แก้ error 
    print("Load build model")
    
    data_lm = load_data(model_path,'model_pkl.pkl')
    data_lm.sanity_check()
    #len(data_lm.train_ds), len(data_lm.valid_ds)

    #print(len(data_lm.train_ds))
    #print(len(data_lm.valid_ds))
    #print(all_df.dtypes)

    #build model
    print("Config model")
    config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False, tie_weights=True, out_bias=True,
                output_p=0.25, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15)
    trn_args = dict(drop_mult=0.9, clip=0.12, alpha=2, beta=1)

    learn = language_model_learner(data_lm, AWD_LSTM, config=config, pretrained=False, **trn_args)

    #load pretrained models
    learn.load_pretrained(**THWIKI_LSTM)

    #train
    print('training unfrozen')
    learn.freeze_to(-1)
    learn.fit_one_cycle(3, 1e-2, moms=(0.8,0.7))

    print("save model")
    learn.save('/home/az2-user/model/wsjson_lm')
    learn.save_encoder('/home/az2-user/model/wsjson_enc')
    learn.load('/home/az2-user/model/wsjson_lm')
    print('finish train')

    print("Load pkl file")
    data_lm = load_data(model_path, "/home/az2-user/model/model_pkl.pkl")
    data_lm.sanity_check()

#classification data
    print("Tokenizer type")
    tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
    processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
                NumericalizeProcessor(vocab=data_lm.vocab, max_vocab=60000, min_freq=2)]#minFq = 20

    data_cls = (ItemLists(model_path,train=TextList.from_df(train_df, model_path, cols=["text"], processor=processor),
                     valid=TextList.from_df(valid_df, model_path, cols=["text"], processor=processor))
    .label_from_df("type")
    .databunch(bs=50)
    )
    data_cls.sanity_check()
    data_cls.save('/home/az2-user/model/data_cls.pkl')
    print("finish build data_cls")
    print(len(data_cls.vocab.itos))

    config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
             output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
    trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

    learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
#load pretrained finetuned model
    learn.load_encoder('/home/az2-user/model/wsjson_enc')
    #train unfrozen
    print("learn freeze type")
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
    learn.unfreeze()
    learn.fit_one_cycle(5, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7),
                            callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name='/home/az2-user/model/model_ml')])
    print("All finish")
    a = learn.predict("สวัสดีครับ")
    print(a[0])
