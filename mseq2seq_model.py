import sys
sys.path.append('utils/')
sys.path.append('seq2seq/')
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core
from model_utils import *
from reward import *
import rlloss
from multi_decoder import dynamic_multi_decode

import time

class Seq2SeqModel():
    def __init__(self, config):

        print('The model is built for training:', config['IS_TRAIN'])

        self.train_mode = 0

        self.learning_rate = tf.Variable(config['LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)
        self.lr_decay_op = self.learning_rate.assign(self.learning_rate * config['LR_DECAY'])
        self.lr_reset_op =  self.learning_rate.assign(config['LR'])

        if config['OPTIMIZER']=='Adam':
            self.optimizer = tf.train.AdamOptimizer
        elif config['OPTIMIZER']=='GD':
            self.optimizer = tf.train.GradientDescentOptimizer
        else:
            raise Exception("Wrong optimizer name...")

        self.global_step = tf.Variable(config['GLOBAL_STEP'], dtype=tf.int32, name='model_global_step', trainable=False)
        self.batch_size = config['BATCH_SIZE']
        self.max_len = config['MAX_OUT_LEN']
        self.input_sizes = config['MODELS_INPUT_VOCAB_SIZES']
        self.output_sizes = config['MODELS_OUTPUT_VOCAB_SIZES']
        self.encoder_hidden_size = config['ENCODER_HIDDEN_SIZE']
        self.decoder_hidden_size = config['DECODER_HIDDEN_SIZE']
        self.embedding_size = config['WORD_EMBEDDING_SIZE']


        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='encoder_inputs_length')
        self.encoder_inputs_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='encoder_inputs_mask')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='decoder_inputs_length')
        self.decoder_inputs_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='decoder_inputs_mask')
        self.decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='decoder_targets_length')
        self.decoder_targets_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='decoder_targets_mask')

        self.maps_g2l_src = []
        self.maps_g2l_tgt = []
        r_global_dict_src, global_dict_src = loadDict(config['SRC_DICT'])
        r_global_dict_tgt, global_dict_tgt = loadDict(config['DST_DICT'])
        def make_maps(dx, dy):
            ret_index = []
            ret_weights = []
            dcnt = {}
            for i in range(len(dx)):
                word = dx[i]
                if word in ['<ASV>','<BBE>','<DARBY>','<DRA>','<WEB>','<YLT>','<AMP>','<CJB>','<CSB>','<ERV>','<ESV>','<KJ21>','<MEV>','<NCV>','<NIV>','<NOG>']:
                    for w in ['<ASV>','<BBE>','<DARBY>','<DRA>','<WEB>','<YLT>','<AMP>','<CJB>','<CSB>','<ERV>','<ESV>','<KJ21>','<MEV>','<NCV>','<NIV>','<NOG>']:
                        if w in dy:
                            word = w
                            print(i, word, dy[word])
                if word not in dy:
                    word = '<UNK>'

                ret_index.append(dy[word])

                if word not in dcnt:
                    dcnt[dy[word]] = 0
                dcnt[dy[word]]+=1
            for index in ret_index:
                ret_weights.append(1.0/dcnt[index])
            return [ret_index, ret_weights]

        for model_no in range(len(config['MODEL_PREFIX'])):
            local_dict, _ = loadDict('auto-train-cc/'+config['MODEL_PREFIX'][model_no]+'/all.in.dict')
            self.maps_g2l_src.append(make_maps(global_dict_src, local_dict))
            local_dict, _ = loadDict('auto-train-cc/'+config['MODEL_PREFIX'][model_no]+'/all.out.dict')
            self.maps_g2l_tgt.append(make_maps(global_dict_tgt, local_dict))

        self.encoder_real_inputs = []
        self.decoder_real_inputs = []
        for model_no in range(len(config['MODEL_PREFIX'])):
            self.encoder_real_inputs.append(tf.gather(self.maps_g2l_src[model_no][0], self.encoder_inputs))
            self.decoder_real_inputs.append(tf.gather(self.maps_g2l_tgt[model_no][0], self.decoder_inputs))

        if config['CORENET']=="FULL":
            self.maps_g2l_src.insert(0,make_maps(global_dict_src, r_global_dict_src))
            self.maps_g2l_tgt.insert(0,make_maps(global_dict_tgt, r_global_dict_tgt))



        self.input_word_embedding_matrixs = []
        self.encoder_inputs_embeddeds = []
        self.encoder_cells = []
        self.encoder_final_outputs = []
        self.encoder_final_states = []
        self.encoder_inputs_length_atts = []

        self.output_word_embedding_matrixs = []
        self.decoder_inputs_embeddeds = []
        self.decoder_cells = []
        self.decoder_initial_states = []
        self.output_projection_layers = []
        self.decoders = []

        self.encoder_variables = []
        self.decoder_variables = []

        self.decoders_outputs = []

        for model_no in range(len(config['MODEL_PREFIX'])):
            model_prefix = "asv-%s-02-"%config['MODEL_PREFIX'][model_no]

            with tf.variable_scope(model_prefix+"DynamicEncoder") as scope:
                self.input_word_embedding_matrixs.append( modelInitWordEmbedding(self.input_sizes[model_no], self.embedding_size, name='we_input'))
                self.encoder_inputs_embeddeds.append( modelGetWordEmbedding(self.input_word_embedding_matrixs[model_no], self.encoder_real_inputs[model_no]))

                self.encoder_cells.append(modelInitRNNCells(self.encoder_hidden_size, config['ENCODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT']))

                if config['BIDIRECTIONAL_ENCODER']:
                    encoder_outputs, encoder_state = modelInitBidirectionalEncoder(self.encoder_cells[model_no], self.encoder_inputs_embeddeds[model_no], self.encoder_inputs_length, encoder_type='stack')
                    self.encoder_final_outputs.append(encoder_outputs)
                    self.encoder_final_states.append(encoder_state)
                else:
                    encoder_outputs, encoder_state = modelInitUndirectionalEncoder(self.encoder_cells[model_no], self.encoder_inputs_embeddeds[model_no], self.encoder_inputs_length)
                    self.encoder_final_outputs.append(encoder_outputs)
                    self.encoder_final_states.append(encoder_state)

                if config['USE_BS'] and not config['IS_TRAIN']:
                    self.encoder_final_states[model_no] = seq2seq.tile_batch(self.encoder_final_states[model_no], config['BEAM_WIDTH'])
                    self.encoder_final_outputs[model_no] = tf.transpose(seq2seq.tile_batch(tf.transpose(self.encoder_final_outputs[model_no], [1,0,2]), config['BEAM_WIDTH']), [1,0,2])

                self.encoder_variables.append(scope.trainable_variables())

            self.encoder_inputs_length_att = self.encoder_inputs_length
            if config['USE_BS'] and not config['IS_TRAIN']:
                self.encoder_inputs_length_att = seq2seq.tile_batch(self.encoder_inputs_length_att, config['BEAM_WIDTH'])

            with tf.variable_scope(model_prefix+"DynamicDecoder") as scope:
                self.output_word_embedding_matrixs.append( modelInitWordEmbedding(self.output_sizes[model_no], self.embedding_size, name='we_output'))
                self.decoder_inputs_embeddeds.append( modelGetWordEmbedding(self.output_word_embedding_matrixs[model_no], self.decoder_real_inputs[model_no]))

                self.decoder_cells.append(modelInitRNNCells(self.decoder_hidden_size, config['DECODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT']))
                if config['ATTENTION_DECODER']:
                    self.decoder_cells[model_no] =  modelInitAttentionDecoderCell(self.decoder_cells[model_no], self.decoder_hidden_size, self.encoder_final_outputs[model_no], self.encoder_inputs_length_att, att_type=config['ATTENTION_MECHANISE'], wrapper_type='whole')
                else:
                    self.decoder_cells[model_no] = modelInitRNNDecoderCell(self.decoder_cells[model_no])

                initial_state = None

                if config['USE_BS'] and not config['IS_TRAIN']:
                    initial_state = self.decoder_cells[model_no].zero_state(batch_size=self.batch_size*config['BEAM_WIDTH'], dtype=tf.float32)
                    if config['ATTENTION_DECODER']:
                        cat_state = tuple([self.encoder_final_states[model_no]] + list(initial_state.cell_state)[:-1])
                        initial_state.clone(cell_state=cat_state)
                    else:
                        initial_state = tuple([self.encoder_final_states[model_no]] + list(initial_state[:-1]))
                else:
                    initial_state = self.decoder_cells[model_no].zero_state(batch_size=self.batch_size, dtype=tf.float32)

                    if config['ATTENTION_DECODER']:
                        cat_state = tuple([self.encoder_final_states[model_no]] + list(initial_state.cell_state)[:-1])
                        initial_state.clone(cell_state=cat_state)
                    else:
                        initial_state = tuple([self.encoder_final_states[model_no]] + list(initial_state[:-1]))

                self.decoder_initial_states.append(initial_state)
                self.output_projection_layers.append( layers_core.Dense(self.output_sizes[model_no], use_bias=False, name=model_prefix+'Opl'))

                decoder_tmp, _ = modelInitPretrainedDecoder(self.decoder_cells[model_no], self.decoder_inputs_embeddeds[model_no], self.decoder_inputs_length,  self.decoder_initial_states[model_no], self.output_projection_layers[model_no])
                self.decoders.append(decoder_tmp)
                # self.decoders_outputs.append(output_tmp)

                self.decoder_variables.append(scope.trainable_variables())



        # print('Encoder Trainable Variables')
        # print(self.encoder_variables)
        # print('Decoder Trainable Variables')
        # print(self.decoder_variables)

        with tf.variable_scope("Core") as scope:
            if config['CORENET']=='FULL':
                self.input_word_embedding_matrixs.insert(0, modelInitWordEmbedding(config['INPUT_VOCAB_SIZE'], self.embedding_size, name='we_input'))
                self.encoder_inputs_embeddeds.insert(0, modelGetWordEmbedding(self.input_word_embedding_matrixs[0], self.encoder_inputs))

                self.encoder_cells.insert(0, modelInitRNNCells(self.encoder_hidden_size, config['ENCODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT']))

                if config['BIDIRECTIONAL_ENCODER']:
                    encoder_outputs, encoder_state = modelInitBidirectionalEncoder(self.encoder_cells[0], self.encoder_inputs_embeddeds[0], self.encoder_inputs_length, encoder_type='stack')
                    self.encoder_final_outputs.insert(0, encoder_outputs)
                    self.encoder_final_states.insert(0, encoder_state)
                else:
                    encoder_outputs, encoder_state = modelInitUndirectionalEncoder(self.encoder_cells[0], self.encoder_inputs_embeddeds[0], self.encoder_inputs_length)
                    self.encoder_final_outputs.insert(0, encoder_outputs)
                    self.encoder_final_states.insert(0, encoder_state)
                self.encoder_inputs_length_att = self.encoder_inputs_length
                self.output_word_embedding_matrixs.insert(0,  modelInitWordEmbedding(config['OUTPUT_VOCAB_SIZE'], self.embedding_size, name='we_output'))
                self.decoder_inputs_embeddeds.insert(0, modelGetWordEmbedding(self.output_word_embedding_matrixs[0], self.decoder_inputs))

                self.decoder_cells.insert(0, modelInitRNNCells(self.decoder_hidden_size, config['DECODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT']))
                if config['ATTENTION_DECODER']:
                    self.decoder_cells[0] =  modelInitAttentionDecoderCell(self.decoder_cells[0], self.decoder_hidden_size, self.encoder_final_outputs[0], self.encoder_inputs_length_att, att_type=config['ATTENTION_MECHANISE'], wrapper_type='whole')
                else:
                    self.decoder_cells[0] = modelInitRNNDecoderCell(self.decoder_cells[0])

                initial_state = self.decoder_cells[0].zero_state(batch_size=self.batch_size, dtype=tf.float32)

                if config['ATTENTION_DECODER']:
                    cat_state = tuple([self.encoder_final_states[0]] + list(initial_state.cell_state)[:-1])
                    initial_state.clone(cell_state=cat_state)
                else:
                    initial_state = tuple([self.encoder_final_states[0]] + list(initial_state[:-1]))

                self.decoder_initial_states.insert(0, initial_state)
                self.output_projection_layers.insert(0, layers_core.Dense(config['OUTPUT_VOCAB_SIZE'], use_bias=False, name='Opl'))

                decoder_tmp, _ = modelInitPretrainedDecoder(self.decoder_cells[0], self.decoder_inputs_embeddeds[0], self.decoder_inputs_length,  self.decoder_initial_states[0], self.output_projection_layers[0])
                self.decoders.insert(0, decoder_tmp)
                # self.decoders_outputs.append(output_tmp)


            self.ma_policy = tf.get_variable(name='ma_policy', shape=[config['OUTPUT_VOCAB_SIZE'],len(config['MODEL_PREFIX'])], dtype=tf.float32)
            print('Core Trainable Variables')
            self.core_variables = scope.trainable_variables()
            print(self.core_variables)


        self.outputs, self.output_weights = modelInitMultiDecodersForTrain(self.decoders, self.ma_policy, self.maps_g2l_tgt, self.output_word_embedding_matrixs, decode_func=dynamic_multi_decode, policy_mode=config['CORENET'])
        if config['IS_TRAIN']:
            self.train_outputs = self.outputs.rnn_output
        if config['USE_BS'] and not config['IS_TRAIN']:
            # self.infer_outputs = modelInitMultiDecodersForBSInfer(self.decoder_cells, [self.decoder_inputs[0]]*len(config[MODEL_PREFIX]), self.output_word_embedding_matrixs, config['BEAM_WIDTH'], config['IDS_END'], config['MAX_OUT_LEN'], self.decoder_initial_states, self.output_projection_layers, self.ma_policy, self.maps_g2l, decode_func=dynamic_multi_decode)
            pass
        else:
            self.infer_outputs = self.outputs.rnn_output

        if config['IS_TRAIN']:
            self.train_loss = seq2seq.sequence_loss(logits=self.train_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask)

            self.rewards = tf.py_func(LMScore, [self.train_outputs, tf.constant(config['LM_MODEL_Y'], dtype=tf.string), tf.constant(config['DST_DICT'], dtype=tf.string)], tf.float32)

            self.train_loss_rl=rlloss.sequence_loss_rl(logits=self.train_outputs, rewards=self.rewards, weights=self.decoder_targets_mask)

            self.eval_loss = seq2seq.sequence_loss(logits=self.train_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask)

            self.final_loss = self.train_loss#+config['RL_RATIO']*self.train_loss_rl



        print('All Trainable Variables:')
        self.all_trainable_variables = self.core_variables
        self.preload_variables = [i+j for i,j in zip(self.encoder_variables, self.decoder_variables)]
        print(self.all_trainable_variables)
        if config['IS_TRAIN']:
            self.train_op = updateBP(self.final_loss, [self.learning_rate], [self.all_trainable_variables], self.optimizer, norm=config['CLIP_NORM'])
            # self.train_op = tf.constant(0.0)
        self.saver = initSaver(tf.global_variables(), config['MAX_TO_KEEP'])
        self.preload_savers = []
        for mno in range(len(config['MODEL_PREFIX'])):
            self.preload_savers.append(initSaver(self.preload_variables[mno]))


    def make_feed(self, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        return {
            self.encoder_inputs: encoder_inputs,
            self.encoder_inputs_length: encoder_inputs_length,
            self.encoder_inputs_mask: encoder_inputs_mask,
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_length,
            self.decoder_inputs_mask: decoder_inputs_mask,
            self.decoder_targets: decoder_targets,
            self.decoder_targets_length: decoder_targets_length,
            self.decoder_targets_mask: decoder_targets_mask,
        }
    def train_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        train_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        ce_loss, rlloss = 0, 0
        [_, ce_loss, rl_loss, weights] = session.run([self.train_op, self.train_loss, self.train_loss_rl, self.output_weights], train_feed)
        # print(weights)
        return [ce_loss, rl_loss]


    def eval_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        [loss, outputs] = session.run([self.eval_loss, self.infer_outputs], infer_feed)
        return loss, outputs
    def test_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        [outputs] = session.run([self.infer_outputs], infer_feed)
        return outputs

def instanceOfInitModel(sess, config):
    ret = Seq2SeqModel(config)
    sess.run(tf.global_variables_initializer())
    mno = 0
    for pf in config['MODEL_PREFIX']:
        pf = 'auto-train-cc/'+pf
        loadModelFromFolder(sess, ret.preload_savers[mno], pf, -1)
        mno += 1
    print('Model Initialized.')
    return ret
