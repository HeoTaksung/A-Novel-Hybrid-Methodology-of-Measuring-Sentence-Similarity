import numpy as np
import tensorflow as tf


def cos_sim(A, B):
    x = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    if np.isnan(x):
        return 0
    return x


def data_load(data, tokenizer, MAX_LEN):
    def bert_tokenizer(sent1, sent2, MAX_LEN):

        encoded_dict = tokenizer.encode_plus(text=sent1, text_pair=sent2, add_special_tokens=True, max_length=MAX_LEN,
											 pad_to_max_length=True, return_attention_mask=True)

        input_id = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        token_type_id = encoded_dict['token_type_ids']

        return input_id, attention_mask, token_type_id

    input_ids = []
    attention_masks = []
    token_type_ids = []
    data_labels = []
    sentence1 = []
    sentence2 = []

    for sent1, sent2, score in data[['Sentence1', 'Sentence2', 'Label']].values:
        try:
            input_id, attention_mask, token_type_id = bert_tokenizer(sent1, sent2, MAX_LEN)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            data_labels.append(float(score) / 5)
            sentence1.append(sent1)
            sentence2.append(sent2)

        except Exception as e:
            print(e)
            print(sent1, sent2)
            pass

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    data_labels = np.array(data_labels, dtype=float)

    inputs = (input_ids, attention_masks, token_type_ids)

    return inputs, data_labels, sentence1, sentence2


def bert_similarity(left, right, model, tokenizer):
    sim = []
    for i in range(len(left)):
        left_sentence = model(tf.constant(tokenizer.encode(left[i])[1:-1])[None, :])
        left_sentence = np.reshape(np.array(left_sentence[1]), (13, np.array(left_sentence[1]).shape[2], 768))
        left_sentence = np.array(left_sentence[1:])
        left_sentence = np.average(left_sentence, axis=0)

        right_sentence = model(tf.constant(tokenizer.encode(right[i])[1:-1])[None, :])
        right_sentence = np.reshape(np.array(right_sentence[1]), (13, np.array(right_sentence[1]).shape[2], 768))
        right_sentence = np.array(right_sentence[1:])
        right_sentence = np.average(right_sentence, axis=0)
        
        left_max_sim = []
        right_max_sim = []
        
        for j in left_sentence:
            etc = []
            for k in right_sentence:
                etc.append(cos_sim(j, k))
            left_max_sim.append(max(etc))
        
        for j in right_sentence:
            etc = []
            for k in left_sentence:
                etc.append(cos_sim(j, k))
            right_max_sim.append(max(etc))
        pairs_sim = (sum(left_max_sim) / len(left_max_sim) + sum(right_max_sim) / len(right_max_sim)) / 2
        
        sim.append(pairs_sim)
    
    return np.array(sim, dtype='float32')
