import tagger


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
    data_iter = tagger.preprocess_date_for_RNN(vectors, batch_size)
    for ((text, text_len)), _ in data_iter:
        print(text)
    print('Yay')


def main():
    # sentences = test_read_training()
    # allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = test_learn_params(sentences)
    # test_baseline_tag_sentence(perWordTagCounts, allTagCounts)
    # # test_viterbi(A, B, sentence=" the small boy")
    # test_hmm_tag_sentence(A, B, sentence=" Jhon likes the blue house at the end of the street")
    test_preprocess()


if __name__ == '__main__':
    main()
