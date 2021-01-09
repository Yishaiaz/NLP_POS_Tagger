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
    vectors = tagger.load_pretrained_embeddings('glove.6B.100d.txt')
    batch_size = 32
    data_iter, pad_index, tag_pad_index, text_field, tags_field = tagger.preprocess_data_for_RNN(vectors, batch_size,
                                                                                                 train_path)

    # after preprocessing
    params_d = {'input_dimension': len(text_field.vocab),
                'embedding_dimension': 100,
                'hidden_dim': 128,
                'output_dimension': len(tags_field.vocab),
                'num_of_layers': 2,
                'dropout': 0.25,
                'pad_idx': pad_index}

    model = tagger.initialize_rnn_model(params_d)
    print("Yay")


def test_train_model():
    tagger.train_model(train_path, 'glove.6B.100d.txt')


def test_evaluate_model():
    tagger.evaluate(test_path)


def test_preprocess_cslstm():
    vectors = tagger.load_pretrained_embeddings('glove.6B.100d.txt')
    batch_size = 32
    data_iter, pad_index, tag_pad_index, text_field, tags_field = tagger.preprocess_data_for_cblstm(vectors, batch_size,
                                                                                                    train_path)
    for (text, text_features, tags), _ in data_iter:
        print(text)
    print('Yay')

def main():
    # sentences = test_read_training()
    # allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = test_learn_params(sentences)
    # test_baseline_tag_sentence(perWordTagCounts, allTagCounts)
    # # test_viterbi(A, B, sentence=" the small boy")
    # test_hmm_tag_sentence(A, B, sentence=" Jhon likes the blue house at the end of the street")
    # test_preprocess()
    # test_init_model()
    # test_train_model()
    # test_evaluate_model()
    test_preprocess_cslstm()


if __name__ == '__main__':
    main()
