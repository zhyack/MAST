# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Seq2seq layer operations for use in neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq import Decoder, BasicDecoderOutput

__all__ = ["dynamic_multi_decode"]


_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access


def _create_zero_outputs(size, dtype, batch_size):
    """Create a zero outputs Tensor structure."""
    def _t(s):
        return (s if isinstance(s, ops.Tensor) else constant_op.constant(
            tensor_shape.TensorShape(s).as_list(),
            dtype=dtypes.int32,
            name="zero_suffix_shape"))

    def _create(s, d):
        return array_ops.zeros(
            array_ops.concat(
                ([batch_size], _t(s)), axis=0), dtype=d)

    return nest.map_structure(_create, size, dtype)


def dynamic_multi_decode(decoders,
                   ma_policy,
                   maps_g2l,
                   word_embeddings,
                   policy_mode=None,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None):
    """Perform dynamic decoding with `decoder`.

    Calls initialize() once and step() repeatedly on the Decoder object.

    Args:
      decoder: A `Decoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major).  If
        `True`, outputs are returned as time major tensors (this mode is faster).
        Otherwise, outputs are returned as batch major tensors (this adds extra
        time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: `int32` scalar, maximum allowed number of decoding
         steps.  Default is `None` (decode until the decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      scope: Optional variable scope to use.

    Returns:
      `(final_outputs, final_state, final_sequence_lengths)`.

    Raises:
      TypeError: if `decoder` is not an instance of `Decoder`.
      ValueError: if `maximum_iterations` is provided but is not a scalar.
    """

    decoders_zero_outputs = []
    final_outputs_ta = None

    def _shape(batch_size, from_shape):
        if not isinstance(from_shape, tensor_shape.TensorShape):
            return tensor_shape.TensorShape(None)
        else:
            batch_size = tensor_util.constant_value(
                ops.convert_to_tensor(
                    batch_size, name="batch_size"))
            return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

    def _create_ta(s, d):
        return tensor_array_ops.TensorArray(
            dtype=d,
            size=0,
            dynamic_size=True,
            element_shape=_shape(decoders[0].batch_size, s))

    def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                  finished, unused_sequence_lengths):
        return math_ops.logical_not(math_ops.reduce_all(finished[0]))

    def local2global(outputs, lno):
        # logits = output_projection_layers[lno](outputs)
        logits = tf.transpose(outputs.rnn_output,[1,0])
        global_logits = tf.transpose(tf.gather(logits, maps_g2l[lno][0]), [1,0])*maps_g2l[lno][1]
        return global_logits
    def global2local(gid, lno):
        lid = tf.gather(maps_g2l[lno][0], gid)
        return tf.nn.embedding_lookup(word_embeddings[lno], lid)

    def body(time, outputs_ta, state, inputs, finished, sequence_lengths):
        """Internal while_loop body.

        Args:
          time: scalar int32 tensor.
          outputs_ta: structure of TensorArray.
          state: (structure of) state tensors and TensorArrays. list
          inputs: (structure of) input tensors. list
          finished: bool tensor (keeping track of what's finished). list
          sequence_lengths: int32 tensor (keeping track of time of finish).

        Returns:
          `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
            next_sequence_lengths)`.
          ```
        """
        decoders_next_outputs = []
        decoders_next_states = []
        decoders_next_inputs = []
        decoders_next_finished = []
        decoders_next_seqlen = []
        decoders_next_ta = []

        outputs_collection = []

        decoder_cnt = 0
        for decoder in decoders:
            (next_outputs, decoder_state, next_inputs, decoder_finished) = decoder.step(time, inputs[decoder_cnt], state[decoder_cnt])
            next_finished = math_ops.logical_or(decoder_finished, finished[decoder_cnt])
            if maximum_iterations is not None:
                next_finished = math_ops.logical_or(
                    next_finished, time + 1 >= maximum_iterations)

            nest.assert_same_structure(state[decoder_cnt], decoder_state)
            nest.assert_same_structure(outputs_ta[decoder_cnt], next_outputs)
            nest.assert_same_structure(inputs[decoder_cnt], next_inputs)
            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(lambda out, zero: array_ops.where(finished[decoder_cnt], zero, out), next_outputs, decoders_zero_outputs[decoder_cnt])
            else:
                emit = next_outputs

            outputs_collection.append(local2global(next_outputs, decoder_cnt))

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else array_ops.where(finished[decoder_cnt], cur, new)

            next_state = None
            if impute_finished:
                next_state = nest.map_structure(
                _maybe_copy_state, decoder_state, state[decoder_cnt])
            else:
                next_state = decoder_state
            this_outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta[decoder_cnt], emit)

            next_sequence_lengths = array_ops.where( math_ops.logical_and(math_ops.logical_not(finished[decoder_cnt]), next_finished), array_ops.fill(array_ops.shape(sequence_lengths[decoder_cnt]), time + 1), sequence_lengths[decoder_cnt])

            decoders_next_inputs.append(next_inputs)
            decoders_next_outputs.append(next_outputs)
            decoders_next_states.append(next_state)
            decoders_next_finished.append(next_finished)
            decoders_next_seqlen.append(next_sequence_lengths)
            decoders_next_ta.append(this_outputs_ta)
            decoder_cnt+=1


        ma_weights = tf.nn.softmax(tf.matmul(outputs_collection[0], ma_policy), -1)
        print('ma_weights_shape:', ma_weights)
        if policy_mode=='FULL':
            outputs_collection = outputs_collection[1:]
        outputs_collection = tf.transpose(ops.convert_to_tensor(outputs_collection, dtype=dtypes.float32),[2,1,0])
        print('all_outputs_shape:', outputs_collection)
        final_outputs=tf.transpose(tf.reduce_sum(outputs_collection*ma_weights, -1), [1,0])
        # final_outputs=tf.transpose(outputs_collection, [2,1,0])[0]
        print('final_outputs_shape:', final_outputs)
        sample_ids = math_ops.cast(math_ops.argmax(final_outputs, axis=-1), dtypes.int32)

        wrapped_final_outputs=BasicDecoderOutput(final_outputs, sample_ids)
        decoders_next_ta.append(nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta[-2], wrapped_final_outputs))
        decoders_next_ta.append(nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta[-1], ma_weights))

        for dno in range(len(decoders)):
            decoders_next_inputs[dno] = global2local(sample_ids, dno)

        outputs_ta = decoders_next_ta
        next_inputs=decoders_next_inputs
        next_state=decoders_next_states
        next_finished=decoders_next_finished
        next_seqlen=decoders_next_seqlen


        return (time + 1, outputs_ta, next_state, next_inputs, next_finished,
                next_seqlen)

    with variable_scope.variable_scope(scope, "decoder") as varscope:
        # Properly cache variable values inside the while_loop
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)
        decoder_cnt = 0
        decoders_initial_finished = []
        decoders_initial_seqlen = []
        decoders_initial_inputs = []
        decoders_initial_state = []
        decoders_outputs_tas = []
        decoders_weights_ta = nest.map_structure(_create_ta, len(decoders), tf.float32)

        if maximum_iterations is not None:
            maximum_iterations = ops.convert_to_tensor(maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")


        initial_time = constant_op.constant(0, dtype=dtypes.int32)



        for decoder in decoders:
            decoder_cnt += 1
            if not isinstance(decoder, Decoder):
                raise TypeError("Expected decoder to be type Decoder_%d, but saw: %s" %(decoder_cnt, type(decoder)))
            initial_finished, initial_inputs, initial_state = decoder.initialize()
            zero_outputs = _create_zero_outputs(decoder.output_size, decoder.output_dtype, decoder.batch_size)
            if maximum_iterations is not None:
                initial_finished = math_ops.logical_or(
                    initial_finished, 0 >= maximum_iterations)
            initial_sequence_lengths = array_ops.zeros_like(
                initial_finished, dtype=dtypes.int32)
            initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size, decoder.output_dtype)

            decoders_initial_finished.append(initial_finished)
            decoders_initial_seqlen.append(initial_sequence_lengths)
            decoders_initial_inputs.append(initial_inputs)
            decoders_initial_state.append(initial_state)
            decoders_zero_outputs.append(zero_outputs)
            decoders_outputs_tas.append(initial_outputs_ta)

        decoders_outputs_tas.append(decoders_outputs_tas[0])
        decoders_outputs_tas.append(decoders_weights_ta)

        res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=[initial_time, decoders_outputs_tas, decoders_initial_state, decoders_initial_inputs, decoders_initial_finished, decoders_initial_seqlen],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        final_outputs_ta = res[1][-2:]
        final_state = res[2][0]
        final_sequence_lengths = res[5]

        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

        # try:
        #     final_outputs, final_state = decoders[0].finalize(
        #         final_outputs, final_state, final_sequence_lengths)
        # except NotImplementedError:
        #     pass

        if not output_time_major:
            final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

    return final_outputs, final_state, final_sequence_lengths
