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
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import decoder
import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


__all__ = [
    "MADecoder",
]


class MADecoder(decoder.Decoder):
  """Multi-Agent sampling decoder."""

  def __init__(self, cell, helper, initial_state, dict_maps, output_layer=None):
    """Initialize BasicDecoder.

    Args:
      cell: Multi(3) `RNNCell` instances.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCells.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._dict_maps = dict_maps
    self._output_layer = output_layer

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        sample_id=tensor_shape.TensorShape([]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        dtypes.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "MADecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = [None,None,None], [None,None,None]
      style_W = self._style_weights
      sw = None
      final_outputs = tf.zeros([self._global_dict_size])
      for i in range(3):
          cell_outputs[i], cell_state[i] = self._cell(inputs, state[i])
          if i == 0:
              sw = tf.matmul(cell_outputs[0], style_W)
          if self._output_layer[i] is not None:
              cell_outputs[i] = self._output_layer(cell_outputs[i])
          if self._dict_maps[i] is not None:
              final_outputs += sw[i]*self.map_global_dict(cell_outputs[i], self._dict_maps[i])
      sample_ids = self._helper.sample(
          time=time, outputs=final_outputs)#, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=final_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(final_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)
