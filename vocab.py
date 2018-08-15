# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
:author: lxm
:description: 字典操作类
:ctime: 2018.07.10 17:44
:mtime: 2018.07.10 17:44
"""

import numpy as np


class Vocab(object):
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        # word
        self.id2word = {}
        self.word2id = {}
        self.word_cnt = {}
        
        # char
        self.id2char = {}
        self.char2id = {}
        self.char_cnt = {}

        self.lower = lower   # lower fn

        self.word_embed_dim = None
        self.word_embeddings = None
        self.char_embed_dim = None
        self.char_embeddings = None

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add_word(token)
            self.add_char(token)

        if filename is not None:
            self.load_from_file(filename)

    def load_from_file(self, file_path):
        for line in open(file_path, 'r'):
            token = line.rstrip('\n')
            self.add_word(token)
            [self.add_char(ctoken) for ctoken in token]

    def word_size(self):
        return len(self.id2word)

    def char_size(self):
        return len(self.id2char)

    def get_word_id(self, token):
        token = token.lower() if self.lower else token
        return self.word2id[token] if token in self.word2id else self.word2id[self.unk_token]

    def get_char_id(self, token):
        token = token.lower() if self.lower else token
        return self.char2id[token] if token in self.char2id else self.char2id[self.unk_token]

    def get_word_token(self, idx):
        return self.id2word[idx] if idx in self.id2word else self.unk_token

    def add_word(self, token, cnt=1):
        token = token.lower() if self.lower else token
        if token in self.word2id:
            idx = self.word2id[token]
        else:
            idx = len(self.id2word)
            self.id2word[idx] = token
            self.word2id[token] = idx
        if cnt > 0:
            if token in self.word_cnt:
                self.word_cnt[token] += cnt
            else:
                self.word_cnt[token] = cnt
        return idx

    def add_char(self, token, cnt=1):
        token = token.lower() if self.lower else token
        if token in self.char2id:
            idx = self.char2id[token]
        else:
            idx = len(self.id2char)
            self.id2char[idx] = token
            self.char2id[token] = idx
        if cnt > 0:
            if token in self.char_cnt:
                self.char_cnt[token] += cnt
            else:
                self.char_cnt[token] = cnt
        return idx

    def filter_words_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.word2id if self.word_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.word2id = {}
        self.id2word = {}
        for token in self.initial_tokens:
            self.add_word(token, cnt=0)

        for token in filtered_tokens:
            self.add_word(token, cnt=0)

    def filter_chars_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.char2id if self.char_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.char2id = {}
        self.id2char = {}
        for token in self.initial_tokens:
            self.add_char(token, cnt=0)
        for token in filtered_tokens:
            self.add_char(token, cnt=0)

    def randomly_init_word_embeddings(self, embed_dim):
        self.word_embed_dim = embed_dim
        self.word_embeddings = np.random.rand(self.word_size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.word_embeddings[self.get_word_id(token)] = np.zeros([self.word_embed_dim])
    
    def randomly_init_char_embeddings(self, embed_dim):
        self.char_embed_dim = embed_dim
        self.char_embeddings = np.random.rand(self.char_size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.char_embeddings[self.get_char_id(token)] = np.zeros([self.char_embed_dim])
    
    """
    :description: for word
    """
    def load_pretrained_word_embeddings(self, embedding_path):
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split()
                token = contents[0].decode('utf8')
                if token not in self.word2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.word_embed_dim is None:
                    self.word_embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.word2id = {}
        self.id2word = {}
        for token in self.initial_tokens:
            self.add_word(token, cnt=0)
        for token in filtered_tokens:
            self.add_word(token, cnt=0)
        # load embeddings
        self.word_embeddings = np.zeros([self.word_size(), self.word_embed_dim])
        for token in self.word2id.keys():
            if token in trained_embeddings:
                self.word_embeddings[self.get_word_id(token)] = trained_embeddings[token]

    def load_pretrained_char_embeddings(self, embedding_path):
        trained_embeddings = {}
        with open(embedding_path, 'rb') as fin:
            word_dict = pkl.load(fin)
            for token in word_dict:
                if token not in self.char2id:
                    continue
                trained_embeddings[token] = word_dict[token]
                if self.char_embed_dim is None:
                    self.char_embed_dim = len(list(trained_embeddings[token]))
        filtered_chars = trained_embeddings.keys()
        # rebuild the token x id map
        self.char2id = {}
        self.id2char = {}
        for char in self.initial_chars:
            self.add_char(char, cnt=0)
        for char in filtered_chars:
            self.add_char(char, cnt=0)

        print("=====unk char id {} , pad char id {}".format(self.char2id[self.unk_char], self.char2id[self.pad_char]))

        if self.unk_token not in self.char2id:
            print("====unknown char is not in char2id=====", self.char_size())
        # load embeddings
        self.char_embeddings = np.random.uniform(low=-0.1, high=0.1, size=(self.char_size(), self.char_embed_dim))
        for char in self.char2id.keys():
            if char in trained_embeddings:
                self.char_embeddings[self.get_char_id(char)] = trained_embeddings[char]

    def convert_word_to_ids(self, tokens):
        vec = [self.get_word_id(label) for label in tokens]
        return vec
    
    def convert_char_to_ids(self, tokens):
        vec = []
        for token in tokens:
            char_vec = []
            for char in token:
                char_vec.append(self.get_char_id(char))
            vec.append(char_vec)
        return vec

    def recover_from_word_ids(self, ids, stop_id=None):
        tokens = []
        for i in ids:
            tokens += [self.get_word_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
