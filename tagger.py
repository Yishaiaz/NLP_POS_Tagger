"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import math
import string
from math import isfinite
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import Vectors
import torch.optim as optim
from collections import Counter
import random
import os

path_separator = os.path.sep

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_word_to_index = {}


# trainpath = 'trainTestData/en-ud-train.upos.tsv'
# testpath = 'trainTestData/en-ud-dev.upos.tsv'


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.


# The expected pipeline for the RNN model:
# Initializing model with all that is needed, including dimensions, training data and pretrained embeddings.
# It is assumed that preprocessing functions will be called from this function. This stage returns an dictionary
# object model_d.
# Training the RNN model: this is done given the output of the initialization model_d and a list of annotated sentences.
# Use the trained model (again, using model_d ) to tag new sentence (the sentence is given as a list of words).
# This is done via the tag_sentence(sentence, model) function.
# Evaluation with count_correct() can be called. Note that this is a general function.

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
    global global_word_to_index
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

    global_word_to_index = perWordTagCounts
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

    end_item = viterbi(sentence, A, B)
    tags = retrace(end_item)
    tagged_sentence = list(map(lambda x: (sentence[x], tags[x]), range(len(tags))))
    return tagged_sentence


def viterbi(sentence, A, B):
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


# a suggestion for a helper function. Not an API requirement
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


# a suggestion for a helper function. Not an API requirement
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
    p = 0  # joint log prob. of words and tags

    end_item = viterbi(sentence, A, B)
    p = end_item[-1]
    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


# ===========================================
#       POS tagging with BiLSTM
# ===========================================

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
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """

    params_d['input_rep'] = params_d.get('input_rep', 0)
    params_d['hidden_dim'] = params_d.get('hidden_dim', 128)
    params_d['dropout'] = params_d.get('dropout', 0.25)

    vectors = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'])
    batch_size = 32

    model = {}
    input_rep = params_d['input_rep']
    train_data = load_annotated_corpus(params_d['data_fn'])
    max_vocab_size = params_d['max_vocab_size']
    min_frequency = params_d['min_frequency']
    if input_rep == 0:
        data_iter, pad_index, tag_pad_index, text_field, tags_field = preprocess_data_for_RNN(vectors, batch_size,
                                                                                              train_data,
                                                                                              max_vocab_size,
                                                                                              min_frequency)
        params_d['input_dimension'] = len(text_field.vocab)
        params_d['output_dimension'] = len(tags_field.vocab)
        params_d['pad_idx'] = pad_index
        params_d['word_to_index'] = text_field.vocab
        params_d['tag_to_index'] = tags_field.vocab

        model = BiLSTMPOSTagger(**params_d)

    elif input_rep == 1:
        data_iter, pad_index, tag_pad_index, text_field, tags_field = preprocess_data_for_cblstm(vectors, batch_size,
                                                                                                 train_data,
                                                                                                 max_vocab_size,
                                                                                                 min_frequency)
        params_d['input_dimension'] = len(text_field.vocab)
        params_d['output_dimension'] = len(tags_field.vocab)
        params_d['pad_idx'] = pad_index
        params_d['word_to_index'] = text_field.vocab
        params_d['tag_to_index'] = tags_field.vocab

        model = CBiLSTMPOSTagger(**params_d)

    return {'lstm': model, 'input_rep': input_rep}


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

    params_d = model.params_d
    return params_d


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    vectors = Vectors(name=path, cache=os.getcwd())
    if vocab is not None:
        vectors = vectors.get_vecs_by_tokens(vocab, True)
    return vectors


def train_rnn(model, train_data, val_data=None, input_rep=0):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    # Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)
    rnn_model = model['lstm']
    params_d = get_model_params(rnn_model)
    input_rep = params_d['input_rep']

    batch_size = 32
    epochs = 10
    vectors = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'])
    max_vocab_size = params_d['max_vocab_size']
    min_frequency = params_d['min_frequency']
    if input_rep == 0:
        data_iter, pad_index, tag_pad_index, text_field, tags_field = preprocess_data_for_RNN(vectors, batch_size,
                                                                                              train_data,
                                                                                              max_vocab_size,
                                                                                              min_frequency)
    else:
        data_iter, pad_index, tag_pad_index, text_field, tags_field = preprocess_data_for_cblstm(vectors, batch_size,
                                                                                                 train_data,
                                                                                                 max_vocab_size,
                                                                                                 min_frequency)

    # set the model embedding
    pretrained_embeddings = text_field.vocab.vectors
    rnn_model.embedding.weight.data.copy_(pretrained_embeddings)

    # set optimizer
    optimizer = optim.Adam(rnn_model.parameters())

    # set criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_index)

    rnn_model = rnn_model.to(device)
    criterion = criterion.to(device)

    features_size = 3
    text_features = []

    rnn_model.train()
    for e in range(epochs):
        epoch_loss = 0
        epoch_acc = 0

        for batch in data_iter:
            text = batch.text
            tags = batch.tags

            if input_rep == 1:
                text_features = batch.text_features - 2
                text_features = text_features.reshape((len(batch), text.shape[-1], features_size))

            optimizer.zero_grad()

            if input_rep == 1:
                predictions = rnn_model(text, text_features)
            else:
                predictions = rnn_model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
        # print('epoch: %s, loss: %s' % (e, epoch_loss/len(data_iter)))

    # torch.save(rnn_model, 'temp.pt')
    if input_rep == 0:
        return {'blstm': [rnn_model, input_rep]}
    else:
        return {'cblstm': [rnn_model, input_rep]}


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
    global global_word_to_index

    input_rep = get_model_params(model)['input_rep']

    model = model.to(device)
    word_to_index = model.word_to_index
    index_to_tag = model.tag_to_index.itos

    features = []
    features_size = 3

    sentence_indices = []
    for word in sentence:
        word_idx = word_to_index[word.lower()]
        if input_rep == 1:
            word_features = word_to_binary(word)
            features.append(word_features)
        sentence_indices.append(word_idx)

    model.eval()

    # input transformations
    sentence_indices = torch.from_numpy(np.array(sentence_indices))

    # cast to tensor int
    sentence_indices = sentence_indices.type('torch.LongTensor')

    # reshape size
    sentence_indices = sentence_indices.reshape(1, sentence_indices.size()[0])

    if input_rep == 1:
        features = torch.from_numpy(np.array(features))
        features = features.type('torch.LongTensor')
        features = features.reshape(1, sentence_indices.shape[-1], features_size)

    if input_rep == 0:
        predictions = model(sentence_indices)
    else:
        predictions = model(sentence_indices, features)

    predictions = predictions.view(-1, predictions.shape[-1])
    max_predictions = predictions.argmax(dim=1)

    tagged_sentence = []
    for i in range(len(sentence)):
        word = sentence[i]
        tag_index = max_predictions[i].item()
        tag = index_to_tag[tag_index]
        tagged_sentence.append((word, tag))

    global_word_to_index = word_to_index
    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    model_params = {'max_vocab_size': float('inf'),
                    'min_frequency': 1,
                    'input_rep': 1,
                    'embedding_dimension': 100,
                    'num_of_layers': 2,
                    'output_dimension': 0,
                    'pretrained_embeddings_fn': 'glove.6B.100d.txt',
                    'data_fn': 'trainTestData' + path_separator + 'en-ud-train.upos.tsv'}
    return model_params


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)

    correct, correctOOV, OOV = 0, 0, 0
    for i in range(len(gold_sentence)):
        word = gold_sentence[i][0]
        gold_tag = gold_sentence[i][1]
        pred_tag = pred_sentence[i][1]

        isCorrect = False
        isOOV = False

        # if word not in global_word_to_index:
        if global_word_to_index[word] == 0 and global_word_to_index[word.lower()] == 0:
            isOOV = True
            OOV += 1

        if pred_tag == gold_tag:
            isCorrect = True
            correct += 1

        if isCorrect and isOOV:
            correctOOV += 1

    return correct, correctOOV, OOV


#  *************************************** added functionality (not api) ***********************************************

def get_possible_tags(word):
    """
    Return all possible word tags
    """
    if word in perWordTagCounts:
        tags_count = perWordTagCounts[word].most_common()
        tags = list(map(lambda x: x[0], tags_count))
    else:
        tags = list(allTagCounts.keys())
    return tags


def build_corpus_text_df(train_tagged_sentences):
    """
    Builds and return a pandas data frame of the given train tagged sentences.
    """
    sentences_and_tags_dicts = []
    for sentence in train_tagged_sentences:
        concat_sen = ''
        concat_tags = ''
        for word, tag in sentence:
            concat_sen += ' ' + word
            concat_tags += ' ' + tag
        temp_dict = {'text': concat_sen, 'tags': concat_tags}
        # temp_dict = {'text': concat_sen}
        sentences_and_tags_dicts.append(temp_dict)

    return pd.DataFrame(sentences_and_tags_dicts)


def preprocess_data_for_RNN(vectors, batch_size, train_tagged_sentences, max_vocab_size, min_frequency):
    """
    preprocess the train tagged sentences, and use BucketIterator for training.
    """
    df = build_corpus_text_df(train_tagged_sentences)
    df.to_csv('train_text_data.csv', index=False)
    text_field = Field(lower=True, batch_first=True)
    tags_field = Field(batch_first=True)

    fields = [('text', text_field), ('tags', tags_field)]
    # TabularDataset

    train_data = TabularDataset(path='train_text_data.csv', format='CSV', fields=fields, skip_header=True)

    # Iterators
    data_iter = BucketIterator(train_data, batch_size=batch_size)

    # Vocabulary
    text_field.build_vocab(train_data, vectors=vectors, min_freq=min_frequency, max_size=max_vocab_size)
    tags_field.build_vocab(train_data, min_freq=min_frequency, max_size=max_vocab_size)

    pad_index = text_field.vocab.stoi[text_field.pad_token]
    tag_pad_index = tags_field.vocab.stoi[tags_field.pad_token]
    return data_iter, pad_index, tag_pad_index, text_field, tags_field


class BiLSTMPOSTagger(nn.Module):
    """
    A class of a vanilla bidirectional lstm
    """
    def __init__(self,
                 input_dimension,
                 max_vocab_size,
                 min_frequency,
                 embedding_dimension,
                 num_of_layers,
                 output_dimension,
                 pretrained_embeddings_fn,
                 data_fn,
                 input_rep,
                 hidden_dim,
                 dropout,
                 pad_idx,
                 word_to_index,
                 tag_to_index):
        super().__init__()

        self.params_d = {'input_dimension': input_dimension,
                         'max_vocab_size': max_vocab_size,
                         'min_frequency': min_frequency,
                         'embedding_dimension': embedding_dimension,
                         'num_of_layers': num_of_layers,
                         'output_dimension': output_dimension,
                         'pretrained_embeddings_fn': pretrained_embeddings_fn,
                         'data_fn': data_fn,
                         'input_rep': input_rep,
                         'hidden_dim': hidden_dim,
                         'dropout': dropout,
                         'pad_idx': pad_idx,
                         'word_to_index': word_to_index,
                         'tag_to_index': tag_to_index}

        self.word_to_index = word_to_index
        self.tag_to_index = tag_to_index
        self.embedding = nn.Embedding(input_dimension, embedding_dimension, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dimension,
                            hidden_dim,
                            batch_first=True,
                            num_layers=num_of_layers,
                            bidirectional=True,
                            dropout=dropout if num_of_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim * 2, output_dimension)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # pass text through embedding layer
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step
        # we use our outputs to make a prediction of what the tag should be
        outputs = self.dropout(outputs)
        predictions = self.fc(outputs)

        return predictions


#  *********************************** CB LSTM added functionality (not api) *******************************************

def word_to_binary(word: str):
    """
    extract word binary features.
    """
    bits = [0 for x in range(3)]
    if word in string.punctuation:
        return bits
    if word.isupper():
        bits[0] = 1
    elif word.capitalize() == word:
        bits[1] = 1
    elif word.islower():
        bits[2] = 1
    return bits


def preprocess_data_for_cblstm(vectors, batch_size, train_tagged_sentences, max_vocab_size, min_frequency):
    """
    preprocess the train tagged sentences, and use BucketIterator for training.
    """
    def transform_to_cs_df(df: pd.DataFrame):
        all_sentences = df.loc[:, ['text']]
        all_cs_bits = list()
        for sentence_idx, sentence in enumerate(all_sentences.values):
            sentence_bits = []
            for word in sentence[0].split():
                sentence_bits = sentence_bits + word_to_binary(word)
            all_cs_bits.append(' '.join([str(x) for x in sentence_bits]))

        cs_df = pd.Series(name='text_features', data=all_cs_bits)
        df['text_features'] = cs_df
        df = df[['text', 'text_features', 'tags']]
        return df

    df = build_corpus_text_df(train_tagged_sentences)
    df = transform_to_cs_df(df)

    df.to_csv('train_text_data.csv', index=False)

    # text_field = Field(tokenize=get_tokenizer("basic_english"), lower=True, include_lengths=True, batch_first=True)
    text_field = Field(lower=True, batch_first=True)
    text_features_field = Field(batch_first=True)
    tags_field = Field(batch_first=True)

    fields = [('text', text_field), ('text_features', text_features_field), ('tags', tags_field)]
    # TabularDataset

    train_data = TabularDataset(path='train_text_data.csv', format='CSV', fields=fields, skip_header=True)

    # Iterators
    data_iter = BucketIterator(train_data, batch_size=batch_size)

    # Vocabulary
    text_field.build_vocab(train_data, vectors=vectors, min_freq=min_frequency, max_size=max_vocab_size)
    tags_field.build_vocab(train_data, min_freq=min_frequency, max_size=max_vocab_size)
    text_features_field.build_vocab(train_data)

    pad_index = text_field.vocab.stoi[text_field.pad_token]
    tag_pad_index = tags_field.vocab.stoi[tags_field.pad_token]
    return data_iter, pad_index, tag_pad_index, text_field, tags_field


class CBiLSTMPOSTagger(nn.Module):
    """
    Case-Based bi-directional LSTM POS tagger.
    uses torch's vanilla lstm model.
    for each word inside the input, calculates and concatenates
    a 3 digit binary vector. the vector is created by the word_to_binary
    function. the possible values for words are:
    1. all caps - creates a vector of 100.
    2. all lower - creates a vector of 001.
    3. first letter is capital case and the rest is lower (title) - 010.

    """
    def __init__(self,
                 input_dimension,
                 max_vocab_size,
                 min_frequency,
                 embedding_dimension,
                 num_of_layers,
                 output_dimension,
                 pretrained_embeddings_fn,
                 data_fn,
                 input_rep,
                 hidden_dim,
                 dropout,
                 pad_idx,
                 word_to_index,
                 tag_to_index):
        super().__init__()

        self.params_d = {'input_dimension': input_dimension,
                         'max_vocab_size': max_vocab_size,
                         'min_frequency': min_frequency,
                         'embedding_dimension': embedding_dimension,
                         'num_of_layers': num_of_layers,
                         'output_dimension': output_dimension,
                         'pretrained_embeddings_fn': pretrained_embeddings_fn,
                         'data_fn': data_fn,
                         'input_rep': input_rep,
                         'hidden_dim': hidden_dim,
                         'dropout': dropout,
                         'pad_idx': pad_idx,
                         'word_to_index': word_to_index,
                         'tag_to_index': tag_to_index}

        self.word_to_index = word_to_index
        self.tag_to_index = tag_to_index
        self.embedding = nn.Embedding(input_dimension, embedding_dimension, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dimension + 3,
                            hidden_dim,
                            batch_first=True,
                            num_layers=num_of_layers,
                            bidirectional=True,
                            dropout=dropout if num_of_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim * 2, output_dimension)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_features):
        # pass text through embedding layer
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        concat = torch.cat((embedded, text_features), 2)

        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(concat)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step
        # we use our outputs to make a prediction of what the tag should be
        outputs = self.dropout(outputs)
        predictions = self.fc(outputs)

        return predictions
