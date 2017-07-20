import os
import sys
import json
import simplejson 
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def predict_unseen_data():
    """Step 0: load trained model and parameters"""
    params = simplejson.loads(open('./parameters.json').read())
    checkpoint_dir = sys.argv[1]
    if not checkpoint_dir.endswith('/'):
        checkpoint_dir += '/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
    logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

    """Step 1: load data for prediction"""
    test_file = sys.argv[2]
    test_examples = simplejson.loads(open(test_file).read())

    # labels.json was saved during training, and it has to be loaded during prediction
    labels = simplejson.loads(open(checkpoint_dir+'/labels.json').read())
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw = [example['text'] for example in test_examples]
    x_test = [data_helper.clean_str(x) for x in x_raw]
    logging.info('The number of x_test: {}'.format(len(x_test)))

    y_test = None
    if 'class' in test_examples[0]:
        y_raw = [example['class'] for example in test_examples]
        y_test = [label_dict[y] for y in y_raw]
        logging.info('The number of y_test: {}'.format(len(y_test)))

    vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_test)))

    """Step 2: compute the predictions"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    if y_test is not None:
        y_test = np.argmax(y_test, axis=1)
        correct_predictions = sum(all_predictions == y_test)
        logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))
        # logging.critical('Predictions are: {}'.format(all_predictions))

def load_model(directory,parametersFile):
    params = simplejson.loads(open('./parameters.json').read())
    if not directory.endswith('/'):
        directory += '/'
    model_file = tf.train.latest_checkpoint(directory + 'checkpoints')
    vocab_path = os.path.join(directory, "vocab.pickle")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    labels = simplejson.loads(open(directory +'labels.json').read())
    # one_hot = np.zeros((len(labels), len(labels)), int)
    # np.fill_diagonal(one_hot, 1)
    # label_dict = dict(zip(labels, one_hot))    
    # print one_hot
    # print labels
    return model_file,vocab_processor,params,labels
    
def predict(model_file,vocab_processor,params,labels,text_list):
    x_test = np.array(list(vocab_processor.transform(text_list)))
    batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
    all_predictions = []
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(model_file))
            saver.restore(sess, model_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
    to_return=[]
    for inc in all_predictions:
        to_return.append(label_dict[int(inc)])
    return to_return

if __name__ == '__main__':
    # python3 predict.py ./trained_model_1478649295/ ./data/small_samples.json
    predict_unseen_data()
    # model_file,vocab_processor,params,label_dict = load_model(sys.argv[1],sys.argv[2])
    # text_list=["and what a goal that was lmfao"]
    # # text_list=["what a goal by mkhi \ud83d\ude4c\ud83c\udfff","mdr un but a 1:42 grosse blague","pogba did awesome for that goal ! awesome","and what a goal that was lmfao","il \u00e9tait beau le but ! # fraisl","le but de ouf \ud83d\ude2e","goal again lol","that goal . . ! #wal #euro 2016","what a goal \ud83d\ude31","giroud ils le attendez ce but # fraisl","baekhyun : what a handsome guy , the jungnang hurricane !","hurricane ! !"]

    # # print label_dict
    # out_pred=predict(model_file,vocab_processor,params,label_dict,text_list)
    # # for inc in out_pred:
        # # print label_dict[int(inc)]
    # print out_pred
    
