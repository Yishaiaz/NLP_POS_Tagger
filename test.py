import tagger

train_path = 'trainTestData/en-ud-train.upos.tsv'
test_path = 'trainTestData/en-ud-dev.upos.tsv'


def test_read_training():
    return tagger.load_annotated_corpus('trainTestData/en-ud-train.upos.tsv')


def test_learn_params(sentences):
    allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(sentences)
    return allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B


def test_baseline_tag_sentence(perWordTagCounts, allTagCounts):
    sentence = "wbnfdc are a winner"
    tagged_sentence = tagger.baseline_tag_sentence(sentence.split(), perWordTagCounts, allTagCounts)
    # print(tagged_sentence)


def test_viterbi(A, B, sentence):
    v_last = tagger.viterbi(sentence.split(), A, B)


def test_hmm_tag_sentence(A, B, sentence):
    tagged_sentence = tagger.hmm_tag_sentence(sentence.split(), A, B)
    print(tagged_sentence)


def test_preprocess():
    vectors = tagger.load_pretrained_embeddings('glove.6B.100d.txt')
    batch_size = 32
    data_iter, pad_index, tag_pad_index, text_field, tags_field = tagger.preprocess_data_for_RNN(vectors, batch_size,
                                                                                                 train_path)
    for ((text, text_len), tags), _ in data_iter:
        print(text)
    print('Yay')


def test_init_model():
    # before preprocessing
    params_d = {'max_vocab_size': float('inf'),
                'min_frequency': 1,
                'input_rep': 1,
                'embedding_dimension': 100,
                'num_of_layers': 2,
                'output_dimension': 0,
                'pretrained_embeddings_fn': 'glove.6B.100d.txt',
                'data_fn': train_path}

    model_d = tagger.initialize_rnn_model(params_d)
    return model_d


def test_train_RNN_model():
    tagger.train_RNN_model(train_path, 'glove.6B.100d.txt')


def test_evaluate_model():
    tagger.evaluate(test_path)


def test_preprocess_cslstm():
    vectors = tagger.load_pretrained_embeddings('glove.6B.100d.txt')
    batch_size = 32
    data_iter, pad_index, tag_pad_index, text_field, tags_field = tagger.preprocess_data_for_cblstm(vectors, batch_size,
                                                                                                    train_path)
    for (text, text_features, tags), _ in data_iter:
        text_features = text_features - 2
        print(text)
    print('Yay')


def test_train_cblstm_model():
    tagger.train_cblstm_model(train_path, 'glove.6B.100d.txt')


def test_evaluate_cblstm_model():
    tagger.evaluate_cblstm(test_path)


def test_api_train_RNN():
    model_d = test_init_model()
    tagger.train_rnn(model_d, tagger.load_annotated_corpus(train_path))


def test_rnn_tag_sentence():
    sentence = "wbnfdc are a Winner"
    tagger.rnn_tag_sentence(model={}, sentence=sentence.split())


def test_rnn_count_correct():
    sentence = "Tamir the AP comes this story :"
    tags = "PROPN DET PROPN VERB DET NOUN PUNCT"
    gold_sentence = [(sentence.split()[i], tags.split()[i]) for i in range(len(sentence.split()))]
    pred_sentence = tagger.rnn_tag_sentence(model={}, sentence=sentence.split())
    correct, correctOOV, OOV = tagger.count_correct(gold_sentence, pred_sentence)
    print(correct, correctOOV, OOV)


def test_baseline_count_correct():
    sentence = "Tamir the AP comes this story :"
    tags = "PROPN DET PROPN VERB DET NOUN PUNCT"
    gold_sentence = [(sentence.split()[i], tags.split()[i]) for i in range(len(sentence.split()))]

    sentences = tagger.load_annotated_corpus('trainTestData/en-ud-train.upos.tsv')
    allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(sentences)
    pred_sentence = tagger.baseline_tag_sentence(sentence.split(), perWordTagCounts, allTagCounts)
    correct, correctOOV, OOV = tagger.count_correct(gold_sentence, pred_sentence)
    print(correct, correctOOV, OOV)


def test_hmm_count_correct():
    sentence = "Tamir the AP comes this story :"
    tags = "PROPN DET PROPN VERB DET NOUN PUNCT"
    gold_sentence = [(sentence.split()[i], tags.split()[i]) for i in range(len(sentence.split()))]

    sentences = tagger.load_annotated_corpus('trainTestData/en-ud-train.upos.tsv')
    allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(sentences)
    pred_sentence = tagger.hmm_tag_sentence(sentence.split(), A, B)
    correct, correctOOV, OOV = tagger.count_correct(gold_sentence, pred_sentence)
    print(correct, correctOOV, OOV)


def test_tag_sentence():
    sentence = "Tamir the AP comes this story :"
    sentences = tagger.load_annotated_corpus('trainTestData/en-ud-train.upos.tsv')
    allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(sentences)

    #  Models that must be supported (you can add more):
    baseline = {'baseline': [perWordTagCounts, allTagCounts]}
    HMM = {'hmm': [A, B]}
    Vanilla_BiLSTM = {'blstm': [{}]}
    BiLSTM_case = {'cblstm': [{}]}

    print(tagger.tag_sentence(sentence.split(), baseline))
    print(tagger.tag_sentence(sentence.split(), HMM))
    print(tagger.tag_sentence(sentence.split(), Vanilla_BiLSTM))
    print(tagger.tag_sentence(sentence.split(), BiLSTM_case))


def main():
    # sentences = test_read_training()
    # allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = test_learn_params(sentences)
    # test_baseline_tag_sentence(perWordTagCounts, allTagCounts)
    # # test_viterbi(A, B, sentence=" the small boy")
    # test_hmm_tag_sentence(A, B, sentence=" Jhon likes the blue house at the end of the street")
    # test_preprocess()
    # test_init_model()
    # test_train_RNN_model()
    # test_evaluate_model()
    # test_preprocess_cslstm()
    # test_train_cblstm_model()
    # test_evaluate_cblstm_model()
    # test_api_train_RNN()
    # test_rnn_tag_sentence()
    # test_rnn_count_correct()
    # test_baseline_count_correct()
    # test_hmm_count_correct()
    test_tag_sentence()


if __name__ == '__main__':
    main()
