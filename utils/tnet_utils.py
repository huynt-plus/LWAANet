import numpy as np
import pickle, os
import string
import codecs

def build_dataset(ds_name, doc_name, bs, dim_w, dim_func):
    dataset, vocab = load_data(ds_name=ds_name, doc_name=doc_name)
    n_train = len(dataset[0])
    n_test = len(dataset[1])
    embeddings = get_embedding(vocab, ds_name, dim_w)
    for i in range(len(embeddings)):
        if i and np.count_nonzero(embeddings[i]) == 0:
            embeddings[i] = np.random.uniform(-0.25, 0.25, embeddings.shape[1])
    embeddings = np.array(embeddings, dtype='float32')
    train_set = pad_dataset(dataset=dataset[0], bs=bs)
    test_set = pad_dataset(dataset=dataset[1], bs=bs)
    embeddings_func = np.random.uniform(-0.25, 0.25, (len(vocab)+1, dim_func))
    embeddings_func[0] = np.zeros(dim_func)
    embeddings_func = np.array(embeddings_func, 'float32')
    return [train_set, test_set], embeddings, embeddings_func, n_train, n_test, vocab

def pad_seq(dataset, field, max_len, symbol):
    """
    pad sequence to max_len with symbol
    """
    n_records = len(dataset)
    for i in range(n_records):
        assert isinstance(dataset[i][field], list)
        while len(dataset[i][field]) < max_len:
            dataset[i][field].append(symbol)
    return dataset

def calculate_position_weight(dataset):
    """
    calculate position weight
    """
    tmax = 40
    ps = []
    n_tuples = len(dataset)
    for i in range(n_tuples):
        dataset[i]['pw'] = []
        weights = []
        for w in dataset[i]['dist']:
            if w == -1:
                weights.append(0.0)
            elif w > tmax:
                weights.append(0.0)
            else:
                weights.append(1.0 - float(w) / tmax)
        #print(weights)
        #ps.append(weights)
        dataset[i]['pw'].extend(weights)
    return dataset

def build_vocab(dataset):
    """
    """
    n_records = len(dataset)
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    idx = len(vocab)
    for i in range(n_records):
        for w in dataset[i]['words']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1

        if dataset[i]['twords'] is not None:
            for w in dataset[i]['twords']:
                if w not in vocab:
                    vocab[w] = idx
                    idx += 1
    return vocab

def get_func_words():
    vocab_func = {}
    # auxiliary verbs, conjunctions, determiners, prepositions, pronouns, quantifiers
    with open('./data/func_words.txt') as fp:
        for line in fp:
            w = line.strip()
            if w not in vocab_func:
                vocab_func[w] = 1
    # add punctuations
    for symbol in string.punctuation:
        vocab_func[symbol] = 1
    return vocab_func

def set_wid(dataset, vocab, max_len):
    """
    word to id
    """
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['words']
        dataset[i]['wids'] = word2id(vocab, sent, max_len)
    return dataset

def set_fid(dataset, vocab, vocab_func, max_len):
    """
    word to functional id
    """
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['words']
        dataset[i]['fids'] = word2fid(vocab, vocab_func, sent, max_len)
    return dataset

def set_tid(dataset, vocab, max_len):
    """
    target word to id
    """
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['twords']
        dataset[i]['tids'] = word2id(vocab, sent, max_len)
    return dataset

def word2id(vocab, sent, max_len):
    """
    mapping word to word id together with sequence padding
    """
    wids = [vocab[w] for w in sent]
    while len(wids) < max_len:
        wids.append(0)
    return wids

def word2fid(vocab, vocab_func, sent, max_len):
    """
    mapping word to function word id
    """
    fids = []
    for w in sent:
        if w in vocab_func:
            fids.append(0)
        else:
            fids.append(vocab[w])
    while len(fids) < max_len:
        fids.append(0)
    return fids

def load_data(ds_name, doc_name):
    """
    """
    if doc_name in ['lt']:
        doc_file = './data/data_doc/electronics_large/text.txt'
    else:
        doc_file = './data/data_doc/yelp_large/text.txt'

    doc_set = read_doc(path=doc_file)

    train_file = './dataset/%s/train.txt' % ds_name
    test_file = './dataset/%s/test.txt' % ds_name
    train_set = read(path=train_file)
    test_set = read(path=test_file)

    train_wc = [t['wc'] for t in train_set]
    test_wc = [t['wc'] for t in test_set]
    max_len = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc)

    train_t_wc = [t['wct'] for t in train_set]
    test_t_wc = [t['wct'] for t in test_set]
    max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)

    #print("maximum length of target:", max_len_target)

    train_set = pad_seq(dataset=train_set, field='dist', max_len=max_len, symbol=-1)
    test_set = pad_seq(dataset=test_set, field='dist', max_len=max_len, symbol=-1)

    # calculate position weight
    train_set = calculate_position_weight(dataset=train_set)
    test_set = calculate_position_weight(dataset=test_set)

    vocab = build_vocab(dataset=train_set+test_set+doc_set)

    vocab_func = get_func_words()

    train_set = set_fid(dataset=train_set, vocab=vocab, vocab_func=vocab_func, max_len=max_len)
    test_set = set_fid(dataset=test_set, vocab=vocab, vocab_func=vocab_func, max_len=max_len)

    train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_len)
    test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_len)

    train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
    test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)

    dataset = [train_set, test_set]

    return dataset, vocab

def get_embedding(vocab, ds_name, dim_w):
    """
    """
    if ds_name == '14semeval_rest':
        emb_file = './vec/glove.42B.300d.txt'   # path of the pre-trained word embeddings
        pkl = './vec/%s_42B.pkl' % ds_name    # word embedding file of the current dataset
    elif ds_name == 'twitter':
        emb_file = './vec/glove.42B.300d.txt'
        pkl = './vec/%s_42B.pkl' % ds_name
    else:
        emb_file = './vec/glove.42B.300d.txt'
        pkl = './vec/%s_42B.pkl' % ds_name
    n_emb = 0
    if os.path.exists(emb_file):
        print("Load embeddings from %s ..." % (emb_file))
        embeddings = np.zeros((len(vocab)+1, dim_w), dtype='float32')
        with open(emb_file) as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                #if embeddings.shape[1] != len(eles[1:]):
                #	embeddings = np.zeros((len(vocab) + 1, len(eles[1:])), dtype='float32')
                if w in vocab:
                    try:
                        embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                        n_emb += 1
                    except ValueError:
                        #print(embeddings[vocab[w]])
                        pass
        print("Find %s word embeddings!!" % n_emb)
        pickle.dump(embeddings, open(pkl, 'wb'))
    else:
        print("Load embeddings from %s ..." % (pkl))
        embeddings = pickle.load(open(pkl, 'rb'))
    return embeddings

def pad_dataset(dataset, bs):
    """
    """
    n_records = len(dataset)
    n_padded = bs - n_records % bs
    new_dataset = [t for t in dataset]
    new_dataset.extend(dataset[:n_padded])
    return new_dataset

def read_doc(path):
    """
    """
    import re
    num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
    def is_number(token):
        return bool(num_regex.match(token))


    dataset = []
    record = {}
    sid = 0
    fin = codecs.open(path, 'r', 'utf-8')
    for line in fin:
        tokens = line.split()
        words = []
        for w in tokens:
            if not is_number(w):
                words.append(w)

        record['sent'] = None
        record['words'] = words.copy()
        record['twords'] = None
        record['wc'] = len(words) # word count
        record['wct'] = None  # target word count
        record['dist'] = None  # relative distance
        record['sid'] = sid
        record['beg'] = None
        record['end'] = None
        # note: if aspect is single word, then aspect
        sid += 1
        dataset.append(record)
    return dataset

def read(path):
    """
    """
    dataset = []
    sid = 0
    with open(path) as fp:
        for line in fp:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            find_label = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    # {pos: 0, neg: 1, neu: 2}
                    # note, this part should be consistent with evals part
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 0
                    elif '/n' in t:
                        end = '/n'
                        y = 1
                    elif '/0' in t:
                        end = '/0'
                        y = 2
                    words.append(t.strip(end))
                    target_words.append(t.strip(end))
                    if not find_label:
                        find_label = True
                        #ys.append(y)
                        record['y'] = y
                        left_most = right_most = tokens.index(t)
                    else:
                        right_most += 1
                else:
                    words.append(t)
            for pos in range(len(tokens)):
                if pos < left_most:
                    d.append(right_most - pos)
                else:
                    d.append(pos - left_most)
            record['sent'] = line.strip()
            record['words'] = words.copy()
            record['twords'] = target_words.copy()  # target words
            #record['twords'] = [target_words[-1]]   # treat the last word as head word
            record['wc'] = len(words)  # word count
            record['wct'] = len(record['twords'])  # target word count
            record['dist'] = d.copy()  # relative distance
            record['sid'] = sid
            record['beg'] = left_most
            record['end'] = right_most + 1
            # note: if aspect is single word, then aspect
            sid += 1
            dataset.append(record)
    return dataset
