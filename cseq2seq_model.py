import sys
sys.path.append('utils/')
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core
from model_utils import *
from reward import *
import rlloss

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
        self.input_size_1 = config['INPUT_VOCAB_SIZE']
        self.input_size_2 = config['OUTPUT_VOCAB_SIZE']
        self.output_size_1 = config['INPUT_VOCAB_SIZE']
        self.output_size_2 = config['OUTPUT_VOCAB_SIZE']
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


        with tf.variable_scope(config['MODEL_PREFIX']+"DynamicEncoder") as scope:
            self.input_word_embedding_matrix_1 = modelInitWordEmbedding(self.input_size_1, self.embedding_size, name='we_input')
            self.encoder_inputs_embedded_1 = modelGetWordEmbedding(self.input_word_embedding_matrix_1, self.encoder_inputs)

            self.encoder_cell_1 = modelInitRNNCells(self.encoder_hidden_size, config['ENCODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT'])
            if config['BIDIRECTIONAL_ENCODER']:
                self.encoder_outputs_1, self.encoder_state_1 = modelInitBidirectionalEncoder(self.encoder_cell_1, self.encoder_inputs_embedded_1, self.encoder_inputs_length, encoder_type='stack')
            else:
                self.encoder_outputs_1, self.encoder_state_1 = modelInitUndirectionalEncoder(self.encoder_cell_1, self.encoder_inputs_embedded_1, self.encoder_inputs_length)

            if config['USE_BS'] and not config['IS_TRAIN']:
                self.encoder_state_1 = seq2seq.tile_batch(self.encoder_state_1, config['BEAM_WIDTH'])
                self.encoder_outputs_1 = tf.transpose(seq2seq.tile_batch(tf.transpose(self.encoder_outputs_1, [1,0,2]), config['BEAM_WIDTH']), [1,0,2])

            # print('Encoder Trainable Variables')
            self.encoder_1_variables = scope.trainable_variables()
            # print(self.encoder_variables)

        with tf.variable_scope(config['MODEL_PREFIX']+"DynamicEncoder_2") as scope:
            self.input_word_embedding_matrix_2 = modelInitWordEmbedding(self.input_size_2, self.embedding_size, name='we_input_2')
            self.encoder_inputs_embedded_2 = modelGetWordEmbedding(self.input_word_embedding_matrix_2, self.encoder_inputs)

            self.encoder_cell_2 = modelInitRNNCells(self.encoder_hidden_size, config['ENCODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT'])
            if config['BIDIRECTIONAL_ENCODER']:
                self.encoder_outputs_2, self.encoder_state_2 = modelInitBidirectionalEncoder(self.encoder_cell_2, self.encoder_inputs_embedded_2, self.encoder_inputs_length, encoder_type='stack')
            else:
                self.encoder_outputs_2, self.encoder_state_2 = modelInitUndirectionalEncoder(self.encoder_cell_2, self.encoder_inputs_embedded_2, self.encoder_inputs_length)

            if config['USE_BS'] and not config['IS_TRAIN']:
                self.encoder_state_2 = seq2seq.tile_batch(self.encoder_state_2, config['BEAM_WIDTH'])
                self.encoder_outputs_2 = tf.transpose(seq2seq.tile_batch(tf.transpose(self.encoder_outputs_2, [1,0,2]), config['BEAM_WIDTH']), [1,0,2])

            # print('Encoder Trainable Variables')
            self.encoder_2_variables = scope.trainable_variables()
            # print(self.encoder_variables)

        self.encoder_inputs_length_att = self.encoder_inputs_length
        if config['USE_BS'] and not config['IS_TRAIN']:
            self.encoder_inputs_length_att = seq2seq.tile_batch(self.encoder_inputs_length_att, config['BEAM_WIDTH'])

        self.encoder_outputs_list = [self.encoder_outputs_1, self.encoder_outputs_2]
        self.encoder_state_list = [self.encoder_state_1, self.encoder_state_2]
        ru = False
        self.train_loss = []
        self.train_loss_rl = []
        self.final_loss = []
        self.eval_loss = []
        self.infer_outputs_all = []
        for mode in range(3):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if ru else None):
                ru = True
                self.encoder_outputs = self.encoder_outputs_list[mode%2]
                self.encoder_state = self.encoder_state_list[mode%2]

                with tf.variable_scope(config['MODEL_PREFIX']+"DynamicDecoder") as scope:
                    self.output_word_embedding_matrix = modelInitWordEmbedding(self.output_size_2, self.embedding_size, name='we_output')
                    self.decoder_inputs_embedded = modelGetWordEmbedding(self.output_word_embedding_matrix, self.decoder_inputs)

                    self.decoder_cell = modelInitRNNCells(self.decoder_hidden_size, config['DECODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT'])
                    if config['ATTENTION_DECODER']:
                        self.decoder_cell = modelInitAttentionDecoderCell(self.decoder_cell, self.decoder_hidden_size, self.encoder_outputs, self.encoder_inputs_length_att, att_type=config['ATTENTION_MECHANISE'], wrapper_type='whole')
                    else:
                        self.decoder_cell = modelInitRNNDecoderCell(self.decoder_cell)


                    initial_state = None

                    if config['USE_BS'] and not config['IS_TRAIN']:
                        initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size*config['BEAM_WIDTH'], dtype=tf.float32)
                        if config['ATTENTION_DECODER']:
                            cat_state = tuple([self.encoder_state] + list(initial_state.cell_state)[:-1])
                            initial_state.clone(cell_state=cat_state)
                        else:
                            initial_state = tuple([self.encoder_state] + list(initial_state[:-1]))
                    else:
                        initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                        if config['ATTENTION_DECODER']:
                            cat_state = tuple([self.encoder_state] + list(initial_state.cell_state)[:-1])
                            initial_state.clone(cell_state=cat_state)
                        else:
                            initial_state = tuple([self.encoder_state] + list(initial_state[:-1]))

                    self.output_projection_layer = layers_core.Dense(self.output_size_2, use_bias=False, name=config['MODEL_PREFIX']+'Opl')
                    if config['IS_TRAIN']:
                        self.train_outputs = modelInitDecoderForTrain(self.decoder_cell, self.decoder_inputs_embedded, self.decoder_inputs_length, initial_state, self.output_projection_layer)
                        self.blind_outputs = modelInitDecoderForBlindTrain(self.decoder_cell, self.decoder_inputs_embedded, self.decoder_inputs_length, self.output_word_embedding_matrix, initial_state, self.output_projection_layer)
                    if config['USE_BS'] and not config['IS_TRAIN']:
                        self.infer_outputs = modelInitDecoderForBSInfer(self.decoder_cell, self.decoder_inputs[0], self.output_word_embedding_matrix, config['BEAM_WIDTH'], config['ID_END_2'], config['MAX_OUT_LEN'], initial_state, self.output_projection_layer)
                    else:
                        self.infer_outputs = modelInitDecoderForGreedyInfer(self.decoder_cell, self.decoder_inputs[0], self.output_word_embedding_matrix, config['ID_END_2'], config['MAX_OUT_LEN'], initial_state, self.output_projection_layer)

                if config['IS_TRAIN']:
                    outputs2use = None
                    self.train_loss.append(seq2seq.sequence_loss(logits=self.train_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask))
                    outputs2use = self.train_outputs
                    if mode==2:
                        self.train_loss[-1] = tf.constant(0.0)
                        outputs2use = self.blind_outputs
                    self.rewards = tf.py_func(LMScore, [outputs2use, tf.constant(config['LM_MODEL_Y'], dtype=tf.string), tf.constant(config['DST_DICT'], dtype=tf.string)], tf.float32)

                    self.train_loss_rl.append(rlloss.sequence_loss_rl(logits=outputs2use, rewards=self.rewards, weights=self.decoder_targets_mask))
                    if not config['RL_ENABLE'] and mode!=2:
                        self.train_loss_rl[-1] = tf.constant(0.0)

                    self.eval_loss.append(seq2seq.sequence_loss(logits=outputs2use, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask))

                    self.final_loss.append(self.train_loss[-1]+config['RL_RATIO']*self.train_loss_rl[-1])
                self.infer_outputs_all.append(self.infer_outputs)

                # print('Decoder Trainable Variables')
                self.decoder_variables = scope.trainable_variables()
                # print(self.decoder_variables)



        print('All Trainable Variables:')
        self.all_trainable_variables = tf.trainable_variables()
        print(self.all_trainable_variables)
        if config['IS_TRAIN']:
            self.train_op = []
            for mode in range(3):
                self.train_op.append(updateBP(self.final_loss[mode], [self.learning_rate], [self.all_trainable_variables], self.optimizer, norm=config['CLIP_NORM']))
        self.saver = initSaver(tf.global_variables(), config['MAX_TO_KEEP'])


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
    def train_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask, mode=0):
        train_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        ce_loss, rlloss = 0, 0
        [_, ce_loss, rl_loss] = session.run([self.train_op[mode], self.train_loss[mode], self.train_loss_rl[mode]], train_feed)
        return [ce_loss, rl_loss]


    def eval_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        [loss, outputs] = session.run([self.eval_loss[0], self.infer_outputs_all[0]], infer_feed)
        return loss, outputs
    def test_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        [outputs] = session.run([self.infer_outputs_all[0]], infer_feed)
        return outputs

def instanceOfInitModel(sess, config):
    ret = Seq2SeqModel(config)
    sess.run(tf.global_variables_initializer())
    print('Model Initialized.')
    return ret
