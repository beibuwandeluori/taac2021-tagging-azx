from . import tokenization
import numpy as np
from transformers import BertTokenizer
import random
import re
from transformers import AutoTokenizer, AutoModel

class TextPreprocess:

    def __init__(self, max_len,
                 vocab='/pubdata/chenby/Tencent/VideoStructuring/MultiModal-Tagging/pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt'):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab)
        self.max_len = max_len

    def __call__(self, text_path):
        with open(text_path, "r") as f:
            text = eval(f.read())
        text = text['video_ocr'] + '|' + text['video_asr']
        text = text[:self.max_len]
        tokens = ['[CLS]']
        for t in text.split("|"):
            tokens += self.tokenizer.tokenize(t)
            tokens += ['[SEP]']
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:self.max_len]
        ids = ids + [0]*(self.max_len-len(ids))
        return np.array(ids).astype('int64')
