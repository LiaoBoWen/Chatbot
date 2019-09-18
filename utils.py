import tensorflow as tf

def convert_id_to_token(inputs,id2token):
    func = lambda x:' '.join(id2token[idx] for idx in inputs)

    return tf.py_func(func,inputs)