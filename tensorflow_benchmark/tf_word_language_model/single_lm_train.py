"""
Entry point for training and eval
"""
import os

import tensorflow as tf
from data_utils import Vocabulary, Dataset
from language_model import LM
from run_utils import run_train, run_eval, run_infer

tf.flags.DEFINE_string("logdir", "lm1b", "Logging directory.")
tf.flags.DEFINE_string("datadir", "1-billion-word-language-modeling-benchmark-r13output", "Logging directory.")
tf.flags.DEFINE_string("mode", "train", "Whether to run 'train' or 'eval' model.")
tf.flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
tf.flags.DEFINE_integer("gpus", 4, "Number of GPUs used.")
tf.flags.DEFINE_integer("eval_steps", 70, "Number of eval steps.")
tf.flags.DEFINE_integer("epochs", 5, "Number of training epochs")

FLAGS = tf.flags.FLAGS

def getOneBillionWordData(data_dir):
    if not os.path.exists(os.path.join(data_dir, "training-monolingual.tokenized.shuffled")):
        os.system("wget -q https://s3.amazonaws.com/one-billion-word-dataset/1-billion-word-language-modeling-benchmark-r13output.tar.gz -P tensorflow_benchmark/tf_word_language_model/")
        os.system("tar -xzvf tensorflow_benchmark/tf_word_language_model/1-billion-word-language-modeling-benchmark-r13output.tar.gz -C tensorflow_benchmark/tf_word_language_model/")
        return "tensorflow_benchmark/tf_word_language_model/1-billion-word-language-modeling-benchmark-r13output"
    else:
        return data_dir

def main(_):
    """
    Start either train or eval. Note hardcoded parts of path for training and eval data
    """
    hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
    hps._set("num_gpus", FLAGS.gpus)
    hps._set("epochs", FLAGS.epochs)
    print('*****HYPER PARAMETERS*****')
    print(hps)
    print('**************************')

    FLAGS.datadir = getOneBillionWordData(FLAGS.datadir)

    vocab = Vocabulary.from_file(os.path.join(FLAGS.datadir, "1b_word_vocab.txt"))

    if FLAGS.mode == "train":
        #hps.batch_size = 256
        dataset = Dataset(vocab, os.path.join(FLAGS.datadir,
                                              "training-monolingual.tokenized.shuffled/*"))
        eval_dataset = Dataset(vocab, os.path.join(FLAGS.datadir,
                                        "heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050"), deterministic=True)
        run_train(dataset, hps, os.path.join(FLAGS.logdir, "train"), ps_device="/gpu:0", eval_dataset=eval_dataset)
    elif FLAGS.mode.startswith("eval_"):
        if FLAGS.mode.startswith("eval_train"):
            data_dir = os.path.join(FLAGS.datadir, "training-monolingual.tokenized.shuffled/*")
        elif FLAGS.mode.startswith("eval_full"):
            data_dir = os.path.join(FLAGS.datadir, "heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050")
        else:
            data_dir = os.path.join(FLAGS.datadir, "heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050")
        dataset = Dataset(vocab, data_dir, deterministic=True)
        run_eval(dataset, hps, FLAGS.logdir, FLAGS.mode, FLAGS.eval_steps)
    elif FLAGS.mode.startswith("infer"):
        data_dir = os.path.join(FLAGS.datadir, "heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050")
        dataset = Dataset(vocab, data_dir, deterministic=True)
        run_infer(dataset, hps, FLAGS.logdir, FLAGS.mode, vocab)


if __name__ == "__main__":
    tf.app.run()
