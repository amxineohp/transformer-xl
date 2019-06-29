import os, gc
import numpy as np
from tfxl import data_utils


def load_data(data_dir, dataset, splits=['train','valid','test'], debug=False):
    corpus = data_utils.get_lm_corpus(os.path.join(data_dir, dataset), dataset)
    train = corpus.train
    valid = corpus.valid
    test = corpus.test
    if debug:
        train = train[0:10001]
        valid = train[0:10001]
    data_dict = {'train':({'seqs':np.expand_dims(train, 0)}, None),
                 'validate': ({'seqs':np.expand_dims(valid, 0)}, None),
                 'test': ({'seqs':np.expand_dims(test, 0)}, None),
                 }
    corpus_info = {
        "vocab_size": len(corpus.vocab),
        "cutoffs": corpus.cutoffs,
    }
    del corpus
    gc.collect()
    return data_dict, corpus_info


