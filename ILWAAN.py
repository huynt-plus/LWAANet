import tensorflow as tf
from utils.utils import get_data_info, load_data, load_word_embeddings, load_bin_vec, get_lex_file_list
from models.ILWAAN_model import ILWAAN_model
from loader.lex_helper import LexHelper
from keras.utils.np_utils import to_categorical
import utils.tnet_utils as tnet_utils
import random

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 100, 'number of example per batch')
    tf.app.flags.DEFINE_integer('n_epoch', 300, 'number of epoch')
    tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
    tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
    tf.app.flags.DEFINE_float('learning_rate', 2e-3, 'learning rate')
    tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
    tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

    tf.app.flags.DEFINE_string('domain', '14semeval_laptop', 'Domain')
    tf.app.flags.DEFINE_string('embedding_fname', './vec/glove.42B.300d.txt', 'embedding file name')
    tf.app.flags.DEFINE_string('embedding', 'word2vec', 'embedding file name')
    tf.app.flags.DEFINE_string('train_fname', './data/restaurant/train.txt', 'training file name')
    tf.app.flags.DEFINE_string('test_fname', './data/restaurant/test.txt', 'testing file name')
    tf.app.flags.DEFINE_string('data_info', './data/data_info.txt', 'the file saving data information')
    tf.app.flags.DEFINE_string('train_data', './data/train_data.txt', 'the file saving training data')
    tf.app.flags.DEFINE_string('test_data', './data/test_data.txt', 'the file saving testing data')
    tf.app.flags.DEFINE_string('lex_path', './resrc/lexicons/lex_config.txt', 'the file saving testing data')
    tf.app.flags.DEFINE_integer('seed', 12345, 'The random seed')
    tf.app.flags.DEFINE_string('lex_embedd_path', './vec/lexicon_embedding', 'the file saving lexicon embedding')
    tf.app.flags.DEFINE_string('checkpoint_path',
                               '',
                               'the file model')

    # tf.set_random_seed(FLAGS.seed)
    # random.seed(FLAGS.seed)

    data = {"word2vec": None, "word2id": None, "max_aspect_len": None, "max_context_len": None, "lex_dim": None}

    if FLAGS.domain == 'twitter' or FLAGS.domain == '14semeval_rest' or FLAGS.domain == '14semeval_laptop':
        dataset, data["word2vec"], embeddings_func, n_train, n_test, data["word2id"] = tnet_utils.build_dataset(
                                                                                        ds_name=FLAGS.domain,
                                                                                        bs=FLAGS.batch_size,
                                                                                        dim_w=300, dim_func=10)
        train_set, test_set = dataset

        train_wc = [t['wc'] for t in train_set]
        test_wc = [t['wc'] for t in test_set]
        data["max_context_len"] = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc)

        train_t_wc = [t['wct'] for t in train_set]
        test_t_wc = [t['wct'] for t in test_set]
        data["max_aspect_len"] = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)

        train_aspects, train_contexts, train_labels, \
        train_aspect_texts, train_context_texts = [x["tids"] for x in train_set], [x["wids"] for x in train_set], \
                                                  [x["y"] for x in train_set], [x["twords"] for x in train_set], \
                                                  [x["words"] for x in train_set]
        train_labels = to_categorical(train_labels, 3)

        train_aspect_lens, train_context_lens = [len(x["twords"]) for x in train_set], [len(x["words"]) for x in
                                                                                        train_set]

        test_aspects, test_contexts, test_labels, \
        test_aspect_texts, test_context_texts = [x["tids"] for x in test_set], [x["wids"] for x in test_set], \
                                                [x["y"] for x in test_set], [x["twords"] for x in test_set], \
                                                [x["words"] for x in test_set]

        test_labels = to_categorical(test_labels, 3)

        test_aspect_lens, test_context_lens = [len(x["twords"]) for x in test_set], \
                                              [len(x["words"]) for x in test_set]

    # Building lexicon embedding
    lex_list = get_lex_file_list(FLAGS.lex_path)
    train = zip(train_aspect_texts, train_context_texts)
    test = zip(test_aspect_texts, test_context_texts)
    lex = LexHelper(lex_list, train, test, max_aspect_len=data["max_aspect_len"],
                    max_context_len=data["max_context_len"])

    if FLAGS.lex_embedd_path == '':
        train_context_lex, train_aspect_lex, test_context_lex, test_aspect_lex, data[
            "lex_dim"] = lex.build_lex_embeddings()
    else:
        train_context_lex, train_aspect_lex, \
        test_context_lex, test_aspect_lex, data["lex_dim"] = lex.load_lexicon_embedding(FLAGS.lex_embedd_path)

    train_data = list(zip(train_aspects, train_contexts, train_labels, train_aspect_lens,
                          train_context_lens, train_aspect_lex, train_context_lex))
    test_data = list(zip(test_aspects, test_contexts, test_labels, test_aspect_lens,
                         test_context_lens, test_aspect_lex, test_context_lex))

    with tf.Session() as sess:
        model = ILWAAN_model(FLAGS, sess, data)
        model.build_model()
        if FLAGS.checkpoint_path == '':
            model.train(train_data, test_data)
        else:
            model.evaluate(test_data)

    print("model=IALSTM_LEX, model_path=%s, embedding=%s, batch-size=%s, n_epoch=%s, n_hidden=%s, domain=%s" % (
        model.out_dir, FLAGS.embedding, FLAGS.batch_size, FLAGS.n_epoch, FLAGS.n_hidden, FLAGS.domain))