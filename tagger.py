"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import math
from math import log, isfinite
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchtext import data
from torchtext.data import Field, get_tokenizer, TabularDataset, BucketIterator
from torchtext.vocab import Vectors
import torch.optim as optim
from collections import Counter

import sys, os, time, platform, nltk, random

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainpath ='trainTestData/en-ud-train.upos.tsv'
testpath = 'trainTestData/en-ud-dev.upos.tsv'

# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=1512021):
    random.seed(seed)
    return seed


SEED = use_seed()
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_deterministic(True)
# torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    # TODO edit the dictionary to have your own details
    # if work is submitted by a pair of students, add the following keys: name2, id2, email2
    return {'name1': 'Tamir Yaffe', 'id1': '305795239', 'email1': 'tamiry@post.bgu.ac.il',
            'name2': 'Yishaia Zabary', 'id2': '307963538', 'email2': 'yishaiaz@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences



START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = Counter()
transitionCounts = Counter()
emissionCounts = Counter()
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transisions probabilities
B = {}  # emmissions probabilities


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and should be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
    tagged_sentences: a list of tagged sentences, each tagged sentence is a
     list of pairs (w,t), as retunred by load_annotated_corpus().

    Return:
    [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    # TODO complete the code
    num_of_sentences = len(tagged_sentences)
    all_possible_tags = []

    for sentence in tagged_sentences:
        prev_tag = START
        for word_tag in sentence:
            word, tag = word_tag
            allTagCounts[tag] += 1
            if perWordTagCounts.get(word) == None:
                perWordTagCounts[word] = Counter()
            if perWordTagCounts[word].get(tag) == None:
                perWordTagCounts[word][tag] = 0
            perWordTagCounts[word][tag] = perWordTagCounts.get((word), {}).get(tag, 0) + 1
            transitionCounts[(prev_tag, tag)] = transitionCounts.get((prev_tag, tag), 0) + 1
            emissionCounts[(tag, word)] = emissionCounts.get((tag, word), 0) + 1
            prev_tag = tag
        transitionCounts[(prev_tag, END)] = transitionCounts.get((prev_tag, END), 0) + 1
    # Calc A & B (Probabilities)
    total_number_of_tags = len(allTagCounts)
    for tag_t in [START] + list(allTagCounts.keys()):
        for tag_t1 in [END] + list(allTagCounts.keys()):
            A[(tag_t, tag_t1)] = transitionCounts.get((tag_t, tag_t1), 1) / (allTagCounts[tag_t] + total_number_of_tags)
    for word in perWordTagCounts.keys():
        for tag in allTagCounts.keys():
            B[(word, tag)] = perWordTagCounts[word].get(tag, 1) / (allTagCounts[tag] + total_number_of_tags)
    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """

    # TODO complete the code
    tagged_sentence = []
    for word in sentence:
        if word in perWordTagCounts:
            tags_counter = perWordTagCounts[word]
            most_common_tag = tags_counter.most_common(1)[0][0]
            word_tag_pair = (word, most_common_tag)
        else:
            tags = list(allTagCounts.keys())
            tag_counts = np.array(list(allTagCounts.values())) / np.array(list(allTagCounts.values())).sum()
            sampled_tag = np.random.choice(tags, p=tag_counts)
            word_tag_pair = (word, sampled_tag)
        tagged_sentence.append(word_tag_pair)

    return tagged_sentence

# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """

    #TODO complete the code
    end_item = viterbi(sentence, A, B)
    tags = retrace(end_item)
    tagged_sentence = list(map(lambda x: (sentence[x], tags[x]), range(len(tags))))
    return tagged_sentence


def viterbi(sentence, A,B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

        """
        # Hint 1: For efficiency reasons - for words seen in training there is no
        #      need to consider all tags in the tagset, but only tags seen with that
        #      word. For OOV you have to consider all tags.
        # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
        #         current list = [ the dummy item ]
        # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END



    # Start with a dummy item  with the START tag (what would it log-prob be?).
    # current list = [ the dummy item ]

    current_tag = START
    prev_best_item = None
    log_prop_so_far = 0  # for summing the log probability

    start_item = (current_tag, prev_best_item, log_prop_so_far)
    viterbi_matrix = [[start_item]]

    for word in sentence:
        possible_tags = get_possible_tags(word)
        col = []
        for tag in possible_tags:
            item = predict_next_best(word, tag, viterbi_matrix[-1], A, B)
            col.append(item)

        viterbi_matrix.append(col)

    # End the sequence with a dummy: the highest-scoring item with the tag END.
    log_prob = list(map(lambda x: x[-1], viterbi_matrix[-1]))
    best_score_item_index = np.argmax(log_prob)
    final_best_score_item = viterbi_matrix[-1][best_score_item_index]
    end_item = (END, final_best_score_item, final_best_score_item[-1])
    viterbi_matrix.append([end_item])

    v_last = end_item
    return v_last


#a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    tags = []
    item = end_item
    tag = item[0]

    while tag is not START:
        if tag is not END:
            tags.append(tag)
        item = item[1]
        tag = item[0]

    return tags[::-1]


#a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list, A, B):
    """Returns a new item (tupple)
    """
    prev_best_item = None
    log_prop_so_far = float('-inf')
    # calculate best prev and log probability

    for prev_item in predecessor_list:
        transition_prob = A[(prev_item[0], tag)]
        emission_prob = B.get((word, tag), 1 / (allTagCounts[tag] + len(allTagCounts)))
        temp_log_prob_so_far = prev_item[-1] + math.log(transition_prob * emission_prob)
        if temp_log_prob_so_far > log_prop_so_far:
            log_prop_so_far = temp_log_prob_so_far
            prev_best_item = prev_item

    return tag, prev_best_item, log_prop_so_far


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): tthe HMM emmission probabilities.
     """
    p = 0   # joint log prob. of words and tags

    #TODO complete the code
    end_item = viterbi(sentence, A, B)
    p = end_item[-1]
    assert isfinite(p) and p<0  # Should be negative. Think why!
    return p


#===========================================
#       POS tagging with BiLSTM
#===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanila biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""

# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)

def initialize_rnn_model(params_d):
    """Returns an lstm model based on the specified parameters.

    Args:
        params_d (dict): an dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'input_dimension': int,
                        'embedding_dimension': int,
                        'num_of_layers': int,
                        'output_dimension': int}
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        torch.nn.Module object
    """

    #TODO complete the code
    model = []
    return model

def get_model_params(model):
    """Returns a dictionary specifying the parameters of the specified model.
    This dictionary should be used to create another instance of the model.

    Args:
        model (torch.nn.Module): the network architecture

    Return:
        a dictionary, containing at least the following keys:
        {'input_dimension': int,
        'embedding_dimension': int,
        'num_of_layers': int,
        output_dimension': int}
    """

    #TODO complete the code
    params_d = {}
    return params_d

def load_pretrained_embeddings(path):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.
    """
    return Vectors(name=path, cache=os.getcwd())


def train_rnn(model, data_fn, pretrained_embeddings_fn):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (torch.nn.Module): the model to train
        data_fn (string): full path to the file with training data (in the provided format)
        pretrained_embeddings_fn (string): full path to the file with pretrained embeddings
    """
    #Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider loading the data and preprocessing it
    # 4. consider using batching
    # 5. some of the above could be implemented in helper functions (not part of
    #    the required API)

    #TODO complete the code
    batch_size = 32
    criterion = nn.CrossEntropyLoss() #you can set the parameters as you like
    vectors = load_pretrained_embeddings(pretrained_embeddings_fn)
    train_iter = preprocess_date_for_RNN(vectors, batch_size)

    model = model.to(device)
    criterion = criterion.to(device)


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence. Tagging is done with the Viterby
        algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (torch.nn.Module):  a trained BiLSTM model

    Return:
        list: list of pairs
    """

    #TODO complete the code
    tagged_sentence = ""
    return tagged_sentence

def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    #TODO complete the code
    model_params = {}
    return model_params


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
        an ordered list of the parameters of the trained model (baseline, HMM)
        or the model itself (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[Torch.nn.Module]}
        4. BiLSTM+case: {'cblstm': [Torch.nn.Module]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the LSTM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.


    Return:
        list: list of pairs
    """
    if model=='baseline':
        return baseline_tag_sentence(sentence, model.values()[0], model.values()[1])
    if model=='hmm':
        return hmm_tag_sentence(sentence, model.values()[0], model.values()[1])
    if model == 'blstm':
        return rnn_tag_sentence(sentence, model.values()[0])
    if model == 'cblstm':
        return rnn_tag_sentence(sentence, model.values()[0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence)==len(pred_sentence)

    #TODO complete the code
    correct, correctOOV, OOV = "", "", ""
    return correct, correctOOV, OOV


# added functions

def get_possible_tags(word):
    if word in perWordTagCounts:
        tags_count = perWordTagCounts[word].most_common()
        tags = list(map(lambda x: x[0], tags_count))
    else:
        tags = list(allTagCounts.keys())
    return tags


def build_corpus_text_df(filename):
    train_tagged_sentences = load_annotated_corpus(filename)
    untagged_sentences = []
    for sentence in train_tagged_sentences:
        concat_sen = ''
        for word, tag in sentence:
            concat_sen += ' ' + word
        untagged_sentences.append(concat_sen)

    return pd.DataFrame({'text':untagged_sentences})


def preprocess_date_for_RNN(vectors, batch_size):
    df = build_corpus_text_df(trainpath)
    df.to_csv('train_text_data.csv', index=False)
    text_field = Field(tokenize=get_tokenizer("basic_english"), lower=True, include_lengths=True, batch_first=True)
    fields = [('text', text_field)]
    # TabularDataset

    data = TabularDataset(path='train_text_data.csv', format='CSV', fields=fields, skip_header=True)

    # Iterators

    data_iter = BucketIterator(data, batch_size=batch_size, sort_key=lambda x: len(x.text),
                               sort=True, sort_within_batch=True)

    # Vocabulary
    # text_field.vocab.load_vectors(vectors)
    text_field.build_vocab(data, vectors=vectors)

    return data_iter

