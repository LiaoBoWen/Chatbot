from data_process import get_batch, load_processed_data, clear_punc, load_vocab
from Model import Transformer
from hyperparams import Params
import tensorflow as tf
import numpy as np
import jieba
import datetime
import time
import os



def train():
    model_params = Params()
    GPU_Option = tf.GPUOptions(per_process_gpu_memory_fraction=model_params.per_process_gpu_memory_fraction)
    session_conf = tf.ConfigProto(
        allow_soft_placement = True,
        log_device_placement = False,
        gpu_options = GPU_Option,
    )

    train_data, test_data = load_processed_data(model_params.processed_data_path)

    print("Test data 's size:{}".format(test_data[0].shape[0]))

    model = Transformer(model_params)
    model.train()

    with tf.device('/gpu:0'):
        with tf.Session(config=session_conf) as sess:
            if not os.path.exists(model_params.summary_path):
                os.mkdir(model_params.summary_path)
            summary_writer = tf.summary.FileWriter(model_params.summary_path,sess.graph)

            if not os.path.exists(model_params.model_save):
                os.mkdir(model_params.model_save)
            saver = tf.train.Saver(max_to_keep=4)
            latest_ckpt = tf.train.latest_checkpoint(model_params.model_save)

            if not latest_ckpt:
                sess.run(tf.global_variables_initializer())
                print('Initial the model .')

            else:
                saver.restore(sess,latest_ckpt)
                print('Restore the model from better_checkpoint .')



            last_loss = 10000
            for xs, decode_inputs, ys, epoch_i in get_batch(train_data,epoch=model_params.epochs,batch_size=model_params.batch_size):
                feed_dict = {model.xs : xs, model.decode_inputs : decode_inputs, model.ys : ys,
                             model.dropout_rate:model_params.dropout_rate}

                _, loss, global_step, y_hat, summary_ = sess.run([model.train_op, model.loss, model.global_step, model.y_hat, model.summary],
                                                       feed_dict=feed_dict)

                summary_writer.add_summary(summary_,global_step)

                if global_step % model_params.print_per_steps == 0:
                    print('{} Epoch: {}, global_step: {}, loss: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %X'),
                                                                           epoch_i + 1, global_step, loss))

                if global_step % model_params.test_per_steps == 0:
                    temp_loss = 0
                    count = 0
                    for xs, decode_inputs, ys, _ in get_batch(test_data,epoch=1,batch_size=model_params.batch_size,shuffle=False):

                        feed_dict = {model.xs : xs, model.decode_inputs : decode_inputs, model.ys : ys,
                                     model.dropout_rate:0}
                        loss = sess.run(model.loss,
                                        feed_dict=feed_dict)
                        temp_loss += loss
                        count += 1

                    loss = temp_loss / count
                    if loss < last_loss:
                        last_loss = loss
                        saver.save(sess,model_params.model_save + '/{}'.format(int(time.time())))
                        print('{}  Save model with lower loss :{}.'.format(datetime.datetime.now().strftime('%Y-%m-%d %X'),
                                                                           loss))

                        # 预览test效果
                        print(decode_inputs[:3])
                        print(ys[:3])
                        print(y_hat[:3])

            summary_writer.close()


def infer():
    def process(inputs):
        inputs = clear_punc(inputs)
        x = []
        for word in jieba.cut(inputs):
            x.append(token2idx.get(word, token2idx['<UNK>']))
        x.append(token2idx['</S>'])
        x = x + [token2idx['<PAD>']] * (model_params.maxlen - len(x))
        x = [x]
        return x

    model_params = Params()
    idx2token, token2idx = load_vocab(model_params.idx2token_path,model_params.token2idx_path)

    model = Transformer(model_params)
    model.eval()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        last_ckpt = tf.train.latest_checkpoint(model_params.model_save)
        saver.restore(sess,last_ckpt)

        while True:
            x = input('{}：>>'.format('笑给我看'))
            #  todo 古诗模式
            if x == '对古诗':
                pass

            x = process(x)
            feed_dict = {model.xs: x}
            y_hat = sess.run(model.y_hat,
                             feed_dict=feed_dict)

            result = ''
            for word in y_hat[0]:
                if word == token2idx['<UNK>']:
                    result += '*××'
                elif word != 3:
                    result += idx2token[word]
                else:
                    break
            if result == '==':
                result = "= ="
            elif result == '<UNK>':
                result = '哎呀，我不知道啊！'


            print('傻逼一号：>>',result,'\n')

if __name__ == '__main__':
    train()
    # infer()