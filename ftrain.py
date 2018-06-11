import sys
sys.path.append('utils/')
from fseq2seq_model import *
from model_utils import *

import argparse
parser = argparse.ArgumentParser(
    description="Train a seq2seq model and save in the specified folder.")
parser.add_argument(
    "-s",
    dest="save_folder",
    type=str,
    default=None,
    help="The specified folder to save. If not specified, the model will not be saved.")
parser.add_argument(
    "-l",
    dest="load_folder",
    type=str,
    help="The specified folder to load saved model. If not specified, the model will be initialized.")
parser.add_argument(
    "-c",
    dest="config_name",
    type=str,
    default=None,
    help="The preset config file name.")
args = parser.parse_args()

CONFIG = dict()
CONFIG['DATASET']=args.config_name

CONFIG['LR'] = 1.0
CONFIG['WE_LR'] = 0.00001
CONFIG['ENCODER_LR'] = 0.00001
CONFIG['DECODER_LR'] = 0.00001
CONFIG['SPLIT_LR'] = False
CONFIG['LR_DECAY'] =  0.7 #0.98
CONFIG['OPTIMIZER'] = 'GD'
CONFIG['CELL'] = "lstm"
CONFIG['WORD_EMBEDDING_SIZE'] = 300
CONFIG['ENCODER_HIDDEN_SIZE'] = 300
CONFIG['DECODER_HIDDEN_SIZE'] = 300
CONFIG['ENCODER_LAYERS'] = 1
CONFIG['DECODER_LAYERS'] = 1
CONFIG['BIDIRECTIONAL_ENCODER'] = True
CONFIG['ATTENTION_DECODER'] = True
# CONFIG['ATTENTION_MECHANISE'] = 'BAHDANAU'
CONFIG['ATTENTION_MECHANISE'] = 'LUONG'
CONFIG['INPUT_DROPOUT'] = 1.0
CONFIG['OUTPUT_DROPOUT'] = 0.7
CONFIG['CLIP']=True
CONFIG['MAX_STEPS_PER_ITER']= 1000
CONFIG['DECAY_STEPS']= 30000
CONFIG['RL_ENABLE']=False
CONFIG['BLEU_RL_ENABLE']=False
CONFIG['RL_RATIO']=0.4
CONFIG['GLOBAL_STEP']=1
CONFIG['CLIP_NORM']=1.0
CONFIG['VAR_NORM_BETA']=0.00003
CONFIG['TRAIN_ON_EACH_STEP']=True
CONFIG['ITERS']=60
CONFIG['MAX_TO_KEEP']=20
CONFIG['BATCH_SIZE']=32

CONFIG['SEED'] = 2333
CONFIG['BUCKETS']=[[85,85]]
CONFIG['USE_BS']=False
CONFIG['BEAM_WIDTH']=10


CONFIG['LOG']=[]
CONFIG['BLEU_LOG']=[]
CONFIG['HYP_FILE_PATH']='data/'
CONFIG['REF_FILE_PATH_FORMAT']='data/'

CONFIG['PRE_ENCODER']=None

global_eval_bleu = locals()['bleuNull']

if CONFIG['DATASET']:
    if os.path.isfile('configs/'+CONFIG['DATASET']+'.json'):
        preset_config = json2load('configs/'+CONFIG['DATASET']+'.json')
        for k in preset_config:
            CONFIG[k] = preset_config[k]
    else:
        raise Exception("Can't find any config file named %s !"%(CONFIG['DATASET']))

may_have_config = loadConfigFromFolder(None, args.load_folder)
if may_have_config:
    preset_config = copy.deepcopy(may_have_config)
    for k in preset_config:
        CONFIG[k] = preset_config[k]

if 'EVAL_BLEU_FUNC' in CONFIG:
    global_eval_bleu = locals()[CONFIG['EVAL_BLEU_FUNC']]


CONFIG['MAX_IN_LEN']=CONFIG['BUCKETS'][-1][0]
CONFIG['MAX_OUT_LEN']=CONFIG['BUCKETS'][-1][1]

def lr_decay_func():
    if CONFIG['GLOBAL_STEP']%CONFIG['DECAY_STEPS']==0:
        CONFIG['LR'] = CONFIG['LR']*CONFIG['LR_DECAY']
        sess.run(Model.lr_decay_op)
        CONFIG['DECAY_STEPS']=CONFIG['MAX_STEPS_PER_ITER']*2

full_dict_src, rev_dict_src = loadDict(CONFIG['SRC_DICT'])
full_dict_dst, rev_dict_dst = loadDict(CONFIG['DST_DICT'])
CONFIG['INPUT_VOCAB_SIZE']=len(rev_dict_src)
CONFIG['OUTPUT_VOCAB_SIZE']=len(rev_dict_dst)
print(CONFIG['INPUT_VOCAB_SIZE'],CONFIG['OUTPUT_VOCAB_SIZE'])
CONFIG['ID_END_2']=full_dict_dst['<EOS>']
CONFIG['ID_BOS_2']=full_dict_dst['<BOS>']
CONFIG['ID_PAD_2']=full_dict_dst['<PAD>']
CONFIG['ID_UNK_2']=full_dict_dst['<UNK>']
CONFIG['ID_END_1']=full_dict_src['<EOS>']
CONFIG['ID_BOS_1']=full_dict_src['<BOS>']
CONFIG['ID_PAD_1']=full_dict_src['<PAD>']
CONFIG['ID_UNK_1']=full_dict_src['<UNK>']


# def findmaxlen(l):
#     ret = 0
#     for i in l:
#         i = i.split()
#         ret = max(ret, len(i))
#     return ret
f_x = codecs.open(CONFIG['TRAIN_X_P'],'r','UTF-8')
x_train_raw_parallel = f_x.readlines()
# print(findmaxlen(x_train_raw))
f_y = codecs.open(CONFIG['TRAIN_Y_P'],'r','UTF-8')
y_train_raw_parallel = f_y.readlines()
# print(findmaxlen(y_train_raw))
train_raw_parallel = [ [x_train_raw_parallel[i].strip(),y_train_raw_parallel[i].strip()] for i in range(len(x_train_raw_parallel))]
train_buckets_raw_parallel = arrangeBuckets(train_raw_parallel, CONFIG['BUCKETS'])
print([len(b) for b in train_buckets_raw_parallel])
f_x.close()
f_y.close()

f_x = codecs.open(CONFIG['TRAIN_X_U'],'r','UTF-8')
x_train_raw_unparallel = f_x.readlines()
y_train_raw_unparallel = x_train_raw_unparallel
train_raw_unparallel = [ [x_train_raw_unparallel[i].strip(),y_train_raw_unparallel[i].strip()] for i in range(len(x_train_raw_unparallel))]
train_buckets_raw_unparallel = arrangeBuckets(train_raw_unparallel, CONFIG['BUCKETS'])
print([len(b) for b in train_buckets_raw_unparallel])
f_x.close()

f_x = codecs.open(CONFIG['TRAIN_Y_DA'],'r','UTF-8')
x_train_raw_dae = f_x.readlines()
f_y = codecs.open(CONFIG['TRAIN_Y_A'],'r','UTF-8')
y_train_raw_dae = f_y.readlines()
train_raw_dae = [ [x_train_raw_dae[i].strip(),y_train_raw_dae[i].strip()] for i in range(len(x_train_raw_dae))]
train_buckets_raw_dae = arrangeBuckets(train_raw_dae, CONFIG['BUCKETS'])
print([len(b) for b in train_buckets_raw_dae])
f_x.close()
f_y.close()

train_buckets_raw = [train_buckets_raw_parallel, train_buckets_raw_dae, train_buckets_raw_unparallel]
dict_train_src = [full_dict_src, full_dict_dst, full_dict_src]
dict_train_dst = [full_dict_dst, full_dict_dst, full_dict_dst]

f_x = codecs.open(CONFIG['DEV_INPUT'],'r','UTF-8')
x_eval_raw = f_x.readlines()
f_y = codecs.open(CONFIG['DEV_OUTPUT'],'r','UTF-8')
y_eval_raw = f_y.readlines()
eval_raw = [ [x_eval_raw[i].strip(),y_eval_raw[i].strip()] for i in range(len(x_eval_raw))]
# eval_raw = eval_raw[:256]
eval_buckets_raw = arrangeBuckets(eval_raw, CONFIG['BUCKETS'])
print([len(b) for b in eval_buckets_raw])
f_x.close()
f_y.close()

CONFIG['RL_RATIO'] = 1.0

g = tf.Graph()
tfconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))
tfconfig.gpu_options.allow_growth=True
with g.as_default():
    tf.set_random_seed(CONFIG['SEED'])
    random.seed(CONFIG['SEED'])
    with tf.Session(config=tfconfig) as sess:
        print('Loading model...')
        CONFIG['IS_TRAIN'] = True
        Model = instanceOfInitModel(sess, CONFIG)
        if args.load_folder != None:
            loadModelFromFolder(sess, Model.saver, args.load_folder, -1)
        tf.set_random_seed(CONFIG['SEED'])
        random.seed(CONFIG['SEED'])
        sess.run(Model.lr_reset_op)
        print('Training Begin...')
        log_losses = CONFIG['LOG']
        log_bleu = CONFIG['BLEU_LOG']
        print(log_losses)
        print(log_bleu)
        all_modes = []
        for mm in CONFIG['MODES']:
            if mm != -1:
                all_modes.append(mm)
        for n_iter in range(CONFIG['GLOBAL_STEP']/CONFIG['MAX_STEPS_PER_ITER'], CONFIG['ITERS']):
            while True:
                b_mode = all_modes[random.randint(0,len(all_modes)-1)]
                b = random.randint(0, min(len(CONFIG['BUCKETS'])-1, n_iter))
                n_b = len(train_buckets_raw[b_mode][b])
                train_batch = [ train_buckets_raw[b_mode][b][random.randint(0, n_b-1)] for _ in range(CONFIG['BATCH_SIZE'])]
                train_batch = map(list, zip(*train_batch))
                model_inputs, len_inputs, inputs_mask = dataSeqs2NpSeqs(train_batch[0], dict_train_src[b_mode], CONFIG['BUCKETS'][b][0])
                model_outputs, len_outputs, outputs_mask = dataSeqs2NpSeqs(train_batch[1], dict_train_dst[b_mode], CONFIG['BUCKETS'][b][1])
                model_targets, len_targets, targets_mask = dataSeqs2NpSeqs(train_batch[1], dict_train_dst[b_mode], CONFIG['BUCKETS'][b][1], bias=1)
                model_input_targets, len_input_targets, input_targets_mask = dataSeqs2NpSeqs(train_batch[0], dict_train_src[b_mode], CONFIG['BUCKETS'][b][0], bias=1)
                batch_loss = Model.train_on_batch(sess, model_inputs, len_inputs, inputs_mask, model_input_targets, len_input_targets, input_targets_mask, model_outputs, len_outputs, outputs_mask, model_targets, len_targets, targets_mask, mode=b_mode)
                print('Mode#%d-Train completed for Iter@%d, Step@%d: CE_Loss=%.6f LM_Loss=%.6f Loss=%.6f LR=%.8f'%(b_mode, n_iter, CONFIG['GLOBAL_STEP'], batch_loss[0], batch_loss[1], batch_loss[0]+batch_loss[1], CONFIG['LR']))
                CONFIG['GLOBAL_STEP']+=1
                lr_decay_func()
                if (CONFIG['GLOBAL_STEP'] % CONFIG['MAX_STEPS_PER_ITER'] == 0):
                    break
            print('Iter@%d completed! Start Evaluating...'%(n_iter))
            eval_losses=[]
            eval_results=dict()
            # eval_buckets_raw = train_buckets_raw
            for b in range(len(CONFIG['BUCKETS'])):
                n_b = len(eval_buckets_raw[b])
                for k in range((n_b+CONFIG['BATCH_SIZE']-1)/CONFIG['BATCH_SIZE']):
                    eval_batch = [ eval_buckets_raw[b][i%n_b] for i in range(k*CONFIG['BATCH_SIZE'], (k+1)*CONFIG['BATCH_SIZE']) ]
                    print('Eval process: [%d/%d] [%d/%d]'%(b+1, len(CONFIG['BUCKETS']), k*CONFIG['BATCH_SIZE'], n_b))
                    eval_batch = map(list, zip(*eval_batch))
                    model_inputs, len_inputs, inputs_mask = dataSeqs2NpSeqs(eval_batch[0], full_dict_src, CONFIG['BUCKETS'][b][0])
                    model_outputs, len_outputs, outputs_mask = dataSeqs2NpSeqs(eval_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1])
                    model_targets, len_targets, targets_mask = dataSeqs2NpSeqs(eval_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1], bias=1)
                    batch_loss, predict_outputs = Model.eval_on_batch(sess, model_inputs, len_inputs, inputs_mask, model_outputs, len_outputs, outputs_mask, model_targets, len_targets, targets_mask)

                    eval_losses.append(batch_loss)
                    eval_batch = map(list, zip(*eval_batch))
                    for i in range(CONFIG['BATCH_SIZE']):
                        eval_results[eval_batch[i][0]] = dataLogits2Seq(predict_outputs[i], rev_dict_dst, calc_argmax=False)
                        if random.random()<0.01:
                            try:
                                print('Raw input: %s\nExpected output: %s\nModel output: %s' % (eval_batch[i][0], eval_batch[i][1], eval_results[eval_batch[i][0]]))
                            except UnicodeDecodeError:
                                pass


            f_x = codecs.open(CONFIG['DEV_INPUT'],'r','UTF-8')
            while(True):
                f_lock = codecs.open('tmp/lock','r','UTF-8')
                l = f_lock.readline().strip()
                f_lock.close()
                if l != 'LOCKED':
                    f_lock = codecs.open('tmp/lock','w','UTF-8')
                    f_lock.write('LOCKED')
                    f_lock.close()
                    break

            f_y = codecs.open('tmp/predictions.txt','w','UTF-8')
            for line in f_x.readlines():
                s = eval_results[line.strip()]
                p = s.find('<EOS>')
                if p==-1:
                    p = len(s)
                f_y.write(s[:p]+'\n')
            f_x.close()
            f_y.close()
            eval_bleu = global_eval_bleu(CONFIG['DEV_OUTPUT'])

            f_lock = codecs.open('tmp/lock','w','UTF-8')
            f_lock.close()

            if eval_bleu == None:
                eval_bleu = 0
            print('Evaluation completed:\nAverage Loss:%.6f\n'%(sum(eval_losses)/len(eval_losses)))
            print('BLEU: ', eval_bleu)

            print(log_losses[max(0,len(log_losses)-100):])
            print(log_bleu[max(0,len(log_losses)-100):])
            log_losses.append(sum(eval_losses)/len(eval_losses))
            log_bleu.append(eval_bleu)
            if log_losses[-1]<=min(log_losses[max(0,len(log_losses)-3):]):
                # sess.run(Model.lr_decay_op)
                print('Learning rate turn down-to %.6f'%(sess.run(Model.learning_rate)))
            if args.save_folder != None:
                CONFIG['LOG']=log_losses
                CONFIG['BLEU_LOG']=log_bleu
                saveModelToFolder(sess, Model.saver, args.save_folder, CONFIG, n_iter)
        print('Training Completed...')
