import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import pickle
from keras.models import Sequential
from keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
import pickle
from keras import losses
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_model(model_name, select_notebook_file):
    # loading the tokenizer
    tokenizer = pickle.load(open(f"{select_notebook_file}/tf_vect.pickle", "rb"))
    model = pickle.load(open(model_name, "rb"))
    return tokenizer, model


def make_pred(tweet, model_name, select_notebook_file):
    total_model_path = f"{select_notebook_file}/{model_name}.pickle"
    tokenizer, model = load_model(total_model_path, select_notebook_file)
    tweet_transform = tokenizer.transform([tweet])
    prediction = model.predict(tweet_transform)
    return prediction


def load_dl_model(model_path, tokenizer_path):
    vocab_size = 5000
    embedding_size = 32
    epochs = 10
    max_len = 50
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation="softmax"))
    model.load_weights(model_path)
    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    return model, tokenizer


def predict_classes(text, model_path, tokenizer_path):
    model, tokenizer = load_dl_model(model_path, tokenizer_path)
    tokenized_text = tokenizer.texts_to_matrix(text)
    padded_text = pad_sequences(tokenized_text, padding="post", maxlen=50)
    health_class = ["Reliable", "Unreliable"]
    prediction = health_class[model.predict(padded_text).argmax(axis=1)[0]]
    return prediction


import tensorflow as tf
import numpy as np
import tf_geometric as tfg

num_classes = 6
from tensorflow import keras
import pickle


class GCNModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn0 = tfg.layers.GCN(100, activation=tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(100, activation=tf.nn.relu)
        self.gcn2 = tfg.layers.GCN(50, activation=tf.nn.relu)
        self.gcn3 = tfg.layers.GCN(num_classes)
        self.dropout = keras.layers.Dropout(0.5)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs
        h = self.gcn0([x, edge_index, edge_weight], cache=cache)
        h = self.gcn1([h, edge_index, edge_weight], cache=cache)
        h = self.dropout(h, training=training)
        h = self.gcn2([h, edge_index, edge_weight], cache=cache)
        h = self.gcn3([h, edge_index, edge_weight], cache=cache)
        return h


def load_gnn_model(tokenizer_path, pmi_model_path):
    tokenizer = pickle.load(open("gnn_tokenizer.pkl", "rb"))
    pmi_model_graph = pickle.load(open("GNN model/cached_pmi_model.p", "rb"))
    embedding_size = 150
    num_words = len(tokenizer.word_index) + 1
    test_graph = build_word_graph(num_words, pmi_model_graph, embedding_size)
    return tokenizer, test_graph


# gnn_model = GCNModel()
class PMIModel(object):
    def __init__(self):
        self.word_counter = None
        self.pair_counter = None

    def get_pair_id(self, word0, word1):
        pair_id = tuple(sorted([word0, word1]))
        return pair_id

    def fit(self, sequences, window_size):
        self.word_counter = Counter()
        self.pair_counter = Counter()
        num_windows = 0
        for sequence in tqdm(sequences):
            for offset in range(len(sequence) - window_size):
                window = sequence[offset : offset + window_size]
                num_windows += 1
                for i, word0 in enumerate(window):
                    self.word_counter[word0] += 1
                    for j, word1 in enumerate(window[i + 1 :]):
                        pair_id = self.get_pair_id(word0, word1)
                        self.pair_counter[pair_id] += 1

        for word, count in self.word_counter.items():
            self.word_counter[word] = count / num_windows
        for pair_id, count in self.pair_counter.items():
            self.pair_counter[pair_id] = count / num_windows

    def transform(self, word0, word1):
        prob_a = self.word_counter[word0]
        prob_b = self.word_counter[word1]
        pair_id = self.get_pair_id(word0, word1)
        prob_pair = self.pair_counter[pair_id]
        if prob_a == 0 or prob_b == 0 or prob_pair == 0:
            return 0

        pmi = np.log(prob_pair / (prob_a * prob_b))
        # print(word0, word1, pmi)
        pmi = np.maximum(pmi, 0.0)
        # print(pmi)
        return pmi


# graph = pickle.load(open("GNN model/cached_pmi_model.p", "rb"))
# tokenizer = pickle.load(open("gnn_tokenizer.pkl", 'rb'))
# tokenized_text = tokenizer.texts_to_matrix("this is a tweet")
# padded_text = pad_sequences(tokenized_text)
def build_combined_graph(word_graph, sequences, embedding_size):
    num_words = word_graph.num_nodes
    x = tf.zeros([len(sequences), embedding_size], dtype=tf.float32)
    edges = []
    edge_weight = []
    for i, sequence in enumerate(sequences):
        doc_node_index = num_words + i
        for word in sequence:
            edges.append([doc_node_index, word])  # only directed edge
            edge_weight.append(1.0)  # use BOW instaead of TF-IDF

    edge_index = np.array(edges).T
    x = tf.concat([word_graph.x, x], axis=0)
    edge_index = np.concatenate([word_graph.edge_index, edge_index], axis=1)
    edge_weight = np.concatenate([word_graph.edge_weight, edge_weight], axis=0)
    return tfg.Graph(x=x, edge_index=edge_index, edge_weight=edge_weight)


def build_word_graph(num_words, pmi_model, embedding_size):
    x = tf.Variable(
        tf.random.truncated_normal(
            [num_words, embedding_size], stddev=1 / np.sqrt(embedding_size)
        ),
        dtype=tf.float32,
    )
    edges = []
    edge_weight = []
    for word0, word1 in pmi_model.pair_counter.keys():
        pmi = pmi_model.transform(word0, word1)
        if pmi > 0:
            edges.append([word0, word1])
            edge_weight.append(pmi)
            edges.append([word1, word0])
            edge_weight.append(pmi)
    edge_index = np.array(edges).T
    return tfg.Graph(x=x, edge_index=edge_index, edge_weight=edge_weight)


def predict_classes(tweet, tokenizer_path, pmi_path):
    tokenizer, test_graph = load_gnn_model(tokenizer_path, pmi_path)
    tokenized_text = tokenizer.texts_to_sequences(["this is a tweet"])
    embedding_size = 150
    num_words = len(tokenizer.word_index) + 1
    gnn_model = GCNModel()
    logits = gnn_model(
        [test_graph.x, test_graph.edge_index, test_graph.edge_weight], training=False
    )
    output = tf.argmax(logits[0], axis=0)
    return output.numpy() % 2
