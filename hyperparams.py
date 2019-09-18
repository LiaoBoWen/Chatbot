
class Params:
    test_per_steps = 100
    print_per_steps = 50
    vocab_size = 32000
    batch_size = 128
    # lr = 0.0003
    lr = 0.00005
    maxlen = 20
    dim_feed_forword =  2048
    num_units = 512
    num_heads = 8
    num_blocks = 6
    dropout_rate = 0.1
    smooth = 0.1
    warmup_step = 4000
    epochs = 50
    per_process_gpu_memory_fraction = 0.7
    vocab_path = '../data/xiaohuangji50w_nofenci.conv'
    processed_data_path = './data/data'
    idx2token_path = './model/idx2token.pkl'
    token2idx_path = './model/token2idx.pkl'
    model_save = './model/checkpoint'
    summary_path = './model/summary'
