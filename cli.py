# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
:author: lxm
:description: 配置文件，包括语料地址，超参数配置
:ctime: 2018.07.10 15:32
:mtime: 2018.07.10 15:32
"""

import tensorflow as tf 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pickle
import logging
import argparse
from dataloader import DataLoader
from vocab import Vocab
from model import Model

def parse_args():
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepro', action='store_true',
                        help='create the directories, prepare to process the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--algo', type=str, default='qanet',
                                help='algorithm')
    train_settings.add_argument('--loss_type', type=str, default='cross_entropy',
                                help='loss fn')
    train_settings.add_argument('--fix_pretrained_vector', type=bool, default=True,
                        help='fixed pretrained vector')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.0001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=1e-5,
                                help='loss weight decay')
    train_settings.add_argument('--decay', type=float, default=None,
                                help='decay')
    train_settings.add_argument('--l2_norm', type=float, default=3e-7,
                                help='l2 norm')
    train_settings.add_argument('--clip_weight', type=bool, default=True,
                                help='clip weight')
    train_settings.add_argument('--max_norm_grad', type=float, default=5.0,
                                help='max norm grad')
    train_settings.add_argument('--dropout', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=16,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--word_embed_size', type=int, default=150,
                                help='size of the word embeddings')
    model_settings.add_argument('--char_embed_size', type=int, default=32,
                                help='size of the char embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=64,
                                help='size of hidden units')
    model_settings.add_argument('--head_size', type=int, default=1, 
                                help='size of head in multihead-attention')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=400,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')
    model_settings.add_argument('--max_ch_len', type=int, default=20,
                                help='max length of character of a word')
    model_settings.add_argument('--use_position_attn', type=bool, default=False,
                                help='use position attention')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/corpus/train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./data/corpus/dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/corpus/test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--save_dir', default='./data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='./data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--pretrained_word_path',default=None,
                               help='path of the log file. If not set, logs are printed to console')
    path_settings.add_argument('--pretrained_char_path',default=None,
                               help='path of the log file. If not set, logs are printed to console')
 
    #path_settings.add_argument('--pretrained_word_path',default="/path/to/Chinese_Word_Vector",
    #                           help='path of the log file. If not set, logs are printed to console')
    #path_settings.add_argument('--pretrained_char_path',default="/path/to/Chinese_Word_Vector",
    #                           help='path of the log file. If not set, logs are printed to console')

    return parser.parse_args()

"""
:description: prepare to process data including building vocab
"""
def prepro(args):
    logger = logging.getLogger("QANet")
    logger.info("====== preprocessing ======")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)

    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    dataloader = DataLoader(args.max_p_num, args.max_p_len, args.max_q_len, args.max_ch_len, 
                          args.train_files, args.dev_files, args.test_files)

    vocab = Vocab(lower=True)
    for word in dataloader.word_iter('train'):
        vocab.add_word(word)
        [vocab.add_char(ch) for ch in word]

    unfiltered_vocab_size = vocab.word_size()
    vocab.filter_words_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.word_size()
    logger.info('After filter {} tokens, the final vocab size is {}, char size is{}'.format(filtered_num,
                                                                            vocab.word_size(), vocab.char_size()))

    unfiltered_vocab_char_size = vocab.char_size()
    vocab.filter_chars_by_cnt(min_cnt=2)
    filtered_char_num = unfiltered_vocab_char_size - vocab.char_size()
    logger.info('After filter {} chars, the final char vocab size is {}'.format(filtered_char_num,
                                                                            vocab.char_size()))

    logger.info('Assigning embeddings...')
    if args.pretrained_word_path is not None:
        vocab.load_pretrained_word_embeddings(args.pretrained_word_path)
    else:
        vocab.randomly_init_word_embeddings(args.word_embed_size)
    
    if args.pretrained_char_path is not None:
        vocab.load_pretrained_char_embeddings(args.pretrained_char_path)
    else:
        vocab.randomly_init_char_embeddings(args.char_embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('====== Done with preparing! ======')

"""
:description: train
"""
def train(args):
    logger = logging.getLogger("QANet")
    logger.info("====== training ======")

    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    dataloader = DataLoader(args.max_p_num, args.max_p_len, args.max_q_len, args.max_ch_len,
                          args.train_files, args.dev_files)

    logger.info('Converting text into ids...')
    dataloader.convert_to_ids(vocab)

    logger.info('Initialize the model...')
    model = Model(vocab, args)

    logger.info('Training the model...')
    model.train(dataloader, args.epochs, args.batch_size, save_dir=args.model_dir, save_prefix=args.algo, dropout=args.dropout)

    logger.info('====== Done with model training! ======')


"""
:descriptoin: evaluate test data
"""
def evaluate(args):
    logger = logging.getLogger("QANet")
    logger.info("====== evaluating ======")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    assert len(args.dev_files) > 0, 'No dev files are provided.'
    dataloader = DataLoader(args.max_p_num, args.max_p_len, args.max_q_len, args.max_ch_len, dev_files=args.dev_files)

    logger.info('Converting text into ids...')
    dataloader.convert_to_ids(vocab)

    logger.info('Restoring the model...')
    model = Model(vocab, args)
    model.restore(args.model_dir, args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = dataloader.next_batch('dev', args.batch_size, vocab.get_word_id(vocab.pad_token), vocab.get_char_id(vocab.pad_token), shuffle=False)

    dev_loss, dev_bleu_rouge = model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')

    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


"""
:descriptoin: predict answers
"""
def predict(args):
    logger = logging.getLogger("QANet")

    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    assert len(args.test_files) > 0, 'No test files are provided.'
    dataloader = DataLoader(args.max_p_num, args.max_p_len, args.max_q_len, args.max_ch_len, 
                          test_files=args.test_files)

    logger.info('Converting text into ids...')
    dataloader.convert_to_ids(vocab)
    logger.info('Restoring the model...')

    model = Model(vocab, args)
    model.restore(args.model_dir, args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = dataloader.next_batch('test', args.batch_size, vocab.get_word_id(vocab.pad_token), vocab.get_char_id(vocab.pad_token), shuffle=False)

    model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    args = parse_args()

    logger = logging.getLogger("QANet")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepro:
        prepro(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
