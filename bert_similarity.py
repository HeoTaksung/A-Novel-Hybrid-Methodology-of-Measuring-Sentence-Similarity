from utils import *
import os
import numpy as np
import tensorflow as tf
from transformers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2"

strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2"])

tf.random.set_seed(1234)
np.random.seed(1234)

BATCH_SIZE = 32
NUM_EPOCHS = 15
MAX_LEN = 256 * 2

tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base", cache_dir='bert_ckpt', do_lower_case=False)

TRAIN_SSE_DF = os.path.join('./', '', 'data/train.csv')
DEV_SSE_DF = os.path.join('./', '', 'data/dev.csv')
TEST_SSE_DF = os.path.join('./', '', 'data/test.csv')

train_data = pd.read_csv(TRAIN_SSE_DF, names=['Sentence1', 'Sentence2', 'Label'])
dev_data = pd.read_csv(DEV_SSE_DF, names=['Sentence1', 'Sentence2', 'Label'])
test_data = pd.read_csv(TEST_SSE_DF, names=['Sentence1', 'Sentence2', 'Label'])

train_inputs, train_data_labels, _, _ = data_load(train_data, tokenizer, MAX_LEN)
dev_inputs, dev_data_labels, dev_sentence1, dev_sentence2 = data_load(dev_data, tokenizer, MAX_LEN)
test_inputs, test_data_labels, test_sentence1, test_sentence2 = data_load(test_data, tokenizer, MAX_LEN)


class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path):
        super(TFBertClassifier, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path, output_hidden_states=True, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), name="classifier")
    
    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        outputs = self.bert(inputs)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits
    
    def save(self):
        return self.bert.save_pretrained('./fine_tuning_model')


with strategy.scope():
    cls_model = TFBertClassifier(model_name="kykim/bert-kor-base", dir_path='bert_ckpt')
    optimizer = tf.keras.optimizers.Adam(3e-5)
    loss = tf.keras.losses.MeanSquaredError()
    cls_model.compile(loss=loss, optimizer=optimizer)

earlystop_callback  = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)

checkpoint_path = os.path.join('./', '', 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))

history = cls_model.fit(train_inputs, train_data_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(dev_inputs, dev_data_labels), callbacks=[earlystop_callback])

y_pred = np.array(cls_model.predict(test_inputs))[:,0]

pearson_r_bert, pearson_p_bert = pearsonr(test_data_labels, y_pred)
spearman_r_bert, spearman_p_bert = spearmanr(test_data_labels, y_pred)
mse = mean_squared_error(test_data_labels, y_pred)

cls_model.save()
model = TFElectraModel.from_pretrained('./fine_tuning_model', output_hidden_states=True)

dev_bert_similar = bert_similarity(dev_sentence1, dev_sentence2, model, tokenizer)
test_bert_similar = bert_similarity(test_sentence1, test_sentence2, model, tokenizer)

y_d = np.array(cls_model.predict(dev_inputs))[:, 0]

c = 0
alpha = 0
for i in range(100, 0, -1):
    t = 0.01 * i
    yy = (1-t) * y_d + t * dev_bert_similar
    pearson_r, pearson_p = pearsonr(dev_data_labels, yy)
    spearman_r, spearman_p = spearmanr(dev_data_labels, yy)
    mse = mean_squared_error(dev_data_labels, yy)
    
    if c < pearson_r:
        c = pearson_r
        alpha = t

yy = (1-alpha) * y_pred + alpha * test_bert_similar
pearson_r, pearson_p = pearsonr(test_data_labels, yy)
spearman_r, spearman_p = spearmanr(test_data_labels, yy)
mse = mean_squared_error(test_data_labels, yy)

print("BERT Pearson : ", pearson_r_bert)
print("BERT Spearman : ", spearman_r_bert)
print("Alpha weight : ", alpha)
print("BERT Word Similarity Pearson : ", pearson_r)
print("BERT Word Similarity Spearman : ", spearman_r)