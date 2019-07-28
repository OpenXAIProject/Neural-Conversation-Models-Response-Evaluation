from config import get_config
from utils import Vocab, get_loader, load_pickle
from models import Solver


def get_data_loader(config, vocab):
    return get_loader(convs=load_pickle(config.convs_path),
                      convs_length=load_pickle(config.conversations_length_path),
                      utterances_length=load_pickle(config.utterances_length_path),
                      vocab=vocab, batch_size=config.batch_size, shuffle=False)


def get_HRED(config, vocab):
    data_loader = get_data_loader(config, vocab)

    return Solver(config, None, data_loader, vocab=vocab, is_train=False)


def main():
    config = get_config(mode='test')

    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    test_solver = get_HRED(config, vocab)

    test_solver.build()
    test_solver.export_samples()


if __name__ == '__main__':
    main()
