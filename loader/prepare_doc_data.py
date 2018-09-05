import numpy as np
import codecs, re

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def create_data(vocab, text_path, label_path, skip_top, skip_len, replace_non_vocab):
    data = []
    label = []  # {pos: 0, neg: 1, neu: 2}
    f = codecs.open(text_path, 'r', 'utf-8')
    f_l = codecs.open(label_path, 'r', 'utf-8')
    num_hit, unk_hit, skip_top_hit, total = 0., 0., 0., 0.
    pos_count, neg_count, neu_count = 0, 0, 0
    max_len = 0

    for line, score in zip(f, f_l):
        word_indices = []
        words = line.split()
        if skip_len > 0 and len(words) > skip_len:
            continue

        score = float(score.strip())
        if score < 3:
            neg_count += 1
            label.append(1)
        elif score > 3:
            pos_count += 1
            label.append(0)
        else:
            neu_count += 1
            label.append(2)

        for word in words:
            if bool(num_regex.match(word)):
                word_indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                word_ind = vocab[word]
                if skip_top > 0 and word_ind < skip_top + 3:
                    skip_top_hit += 1
                else:
                    word_indices.append(word_ind)
            else:
                if replace_non_vocab:
                    word_indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        if len(word_indices) > max_len:
            max_len = len(word_indices)

        data.append(word_indices)

    f.close()
    f_l.close()

    final_data = []
    for indices in data:
        while len(indices) < max_len:
            indices.append(0)
        final_data.append(indices)

    print('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))

    return np.array(final_data), np.array(label), max_len

def prepare_data_doc(vocab, domain, skip_top=0, skip_len=0, replace_non_vocab=1):
    print("Loading %s" % domain)
    if domain in ['electronics_large']:
        text_path = './data/data_doc/electronics_large/text.txt'
        score_path = './data/data_doc/electronics_large/label.txt'
    else:
        text_path = './data/data_doc/yelp_large/text.txt'
        score_path = './data/data_doc/yelp_large/label.txt'

    data, label, max_len = create_data(vocab, text_path, score_path, skip_top, skip_len, replace_non_vocab)

    return data, label, max_len

def prepare_data(domain, vocab, ds_name, dim_w, is_load_embedding=False):

    doc_data, doc_label, doc_maxlen = prepare_data_doc(vocab, domain)

    if is_load_embedding:
        embeddings = get_embedding(vocab, ds_name, dim_w)
        for i in range(len(embeddings)):
            if i and np.count_nonzero(embeddings[i]) == 0:
                embeddings[i] = np.random.uniform(-0.25, 0.25, embeddings.shape[1])
        embeddings = np.array(embeddings, dtype='float32')

    return doc_data, doc_label, doc_maxlen, embeddings

def get_embedding(vocab, ds_name, dim_w):
    import os

    emb_file = None
    embeddings = None

    if ds_name == 'yelp_large':
        emb_file = './vec/yelp_large.300d.txt'
    elif ds_name == 'electronics_large':
        emb_file = './vec/electronics_large.txt'

    print("Load embeddings from %s" % (emb_file))
    n_emb = 0
    if os.path.exists(emb_file):
        embeddings = np.zeros((len(vocab)+1, dim_w), dtype='float32')
        with open(emb_file) as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                #if embeddings.shape[1] != len(eles[1:]):
                #	embeddings = np.zeros((len(vocab) + 1, len(eles[1:])), dtype='float32')
                n_emb += 1
                if w in vocab:
                    try:
                        embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                    except ValueError:
                        #print(embeddings[vocab[w]])
                        pass
        print("Find %s word embeddings!!" % n_emb)

    return embeddings

