# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Class CollectiveAllReduceStrategy implementing DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import weakref

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)

def _push_pull(op, tensor, scope='', name=None):
    """An op which sums an input tensor over all the BytePS processes.
    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all BytePS processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.
    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
    """
    if name is None and not _executing_eagerly():
        name = 'BytePSPushPull_%s' % _normalize_name(tensor.name)
    if scope == '' and not _executing_eagerly():
        if 'v1' in dir(tf.compat):
            scope = tf.compat.v1.get_default_graph().get_name_scope()
        else:
            scope = tf.get_default_graph().get_name_scope()
        if scope != '':
            scope += '/'
    full_name = scope + name
    full_name = full_name.encode("ascii")
    TF_LIB_CTYPES.byteps_tensorflow_declare_tensor(ctypes.c_char_p(full_name))
    return C_LIB.byteps_push_pull(tensor, name=name)

# TODO(yuefengz): support in-graph replication.
@tf_export("byteps.MultiWorkerMirroredStrategy", v1=[])
class CollectiveAllReduceStrategy(distribute_lib.Strategy):
  """A distribution strategy for synchronous training on multiple workers.

  This strategy implements synchronous distributed training across multiple
  workers, each with potentially multiple GPUs. Similar to
  `tf.distribute.MirroredStrategy`, it creates copies of all variables in the
  model on each device across all workers.

  It uses CollectiveOps's implementation of multi-worker all-reduce to
  to keep variables in sync. A collective op is a single op in the
  TensorFlow graph which can automatically choose an all-reduce algorithm in
  the TensorFlow runtime according to hardware, network topology and tensor
  sizes.

  By default it uses all local GPUs or CPU for single-worker training.

  When 'TF_CONFIG' environment variable is set, it parses cluster_spec,
  task_type and task_id from 'TF_CONFIG' and turns into a multi-worker strategy
  which mirrored models on GPUs of all machines in a cluster. In the current
  implementation, it uses all GPUs in a cluster and it assumes all workers have
  the same number of GPUs.

  You can also pass a `distribute.cluster_resolver.ClusterResolver` instance
  when instantiating the strategy. The task_type, task_id etc. will be parsed
  from the resolver instance instead of from the `TF_CONFIG` env var.

  It supports both eager mode and graph mode. However, for eager mode, it has to
  set up the eager context in its constructor and therefore all ops in eager
  mode have to run after the strategy object is created.

  """
  # TODO(anjalisridhar): Update our guides with examples showing how we can use
  # the cluster_resolver argument.

  def __init__(
      self,
      communication=cross_device_ops_lib.CollectiveCommunication.AUTO,
      cluster_resolver=None):
    """Creates the strategy.

    Args:
      communication: optional Enum of type
        `distribute.experimental.CollectiveCommunication`.  This provides a way
        for the user to override the choice of collective op communication.
        Possible values include `AUTO`, `RING`, and `NCCL`.
      cluster_resolver: optional `distribute.cluster_resolver.ClusterResolver`
        object. The default ClusterResolver that is used is the
        TFConfigClusterResolver which is instantiated from the TF_CONFIG env
        var.
    """
    # TODO(b/150151677): consider move communication to CollectiveHints.
    super(CollectiveAllReduceStrategy, self).__init__(
        MyCollectiveAllReduceExtended(
            self,
            communication=communication,
            cluster_resolver=cluster_resolver))

    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "MultiWorkerMirroredStrategy")
    # pylint: disable=protected-access
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended._num_gpus_per_worker)

  @classmethod
  def _from_local_devices(cls, devices):
    """A convenience method to create an object with a list of devices."""
    obj = cls()
    obj.extended._initialize_local(TFConfigClusterResolver(), devices=devices)  # pylint: disable=protected-access
    return obj

  def scope(self):  # pylint: disable=useless-super-delegation
    """Returns a context manager selecting this Strategy as current.

    Inside a `with strategy.scope():` code block, this thread
    will use a variable creator set by `strategy`, and will
    enter its "cross-replica context".

    In `MultiWorkerMirroredStrategy`, all variables created inside
    `strategy.scope() will be mirrored on all replicas of each worker.
    Moreover, it also sets a default device scope so that ops without
    specified devices will end up on the correct worker.

    Returns:
      A context manager to use for creating variables with this strategy.
    """
    return super(CollectiveAllReduceStrategy, self).scope()


@tf_export(v1=["distribute.experimental.MultiWorkerMirroredStrategy"])  # pylint: disable=missing-docstring
class CollectiveAllReduceStrategyV1(distribute_lib.StrategyV1):

  __doc__ = CollectiveAllReduceStrategy.__doc__

  def __init__(
      self,
      communication=cross_device_ops_lib.CollectiveCommunication.AUTO,
      cluster_resolver=None):
    """Initializes the object."""
    super(CollectiveAllReduceStrategyV1, self).__init__(
        CollectiveAllReduceExtended(
            self,
            communication=communication,
            cluster_resolver=cluster_resolver))
    distribute_lib.distribution_strategy_gauge.get_cell("V1").set(
        "MultiWorkerMirroredStrategy")
    # pylint: disable=protected-access
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_gpu_per_worker").set(self.extended._num_gpus_per_worker)


class MyCollectiveAllReduceExtended(CollectiveAllReduceExtended):
  """Implementation of CollectiveAllReduceStrategy."""

  def __init__(self,
               container_strategy,
               communication,
               cluster_resolver):
    super(MyCollectiveAllReduceExtended, self).__init__
               container_strategy,
               communication,
               cluster_resolver):

  def _initialize_strategy(self, cluster_resolver):
    if cluster_resolver.cluster_spec().as_dict():
      self._initialize_multi_worker(cluster_resolver)
    else:
      self._initialize_local(cluster_resolver)

  def _reduce_to(self, reduce_op, value, destinations, experimental_hints):
    if (isinstance(value, values.Mirrored) and
        reduce_op == reduce_util.ReduceOp.MEAN):
      return value
    assert not isinstance(value, values.Mirrored)

    if (isinstance(value, values.DistributedValues) and
        len(self.worker_devices) == 1):
      value = value.values[0]

    # When there are multiple workers, we need to reduce across workers using
    # collective ops.
    if (not isinstance(value, values.DistributedValues) and
        self._num_workers == 1):
      # This function handles reducing values that are not PerReplica or
      # Mirrored values. For example, the same value could be present on all
      # replicas in which case `value` would be a single value or value could
      # be 0.
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, value, destinations, len(self.worker_devices))
    # replace this reduce() call to push_pull
    return _push_pull(reduce_op, value)

# destinations: the reduction destinations.
#    return self._get_cross_device_ops().reduce(
#        reduce_op,
#        value,
#        destinations=destinations,
#        experimental_hints=experimental_hints)

  def _warn_nccl_no_gpu(self):
    if ((self._communication ==
         cross_device_ops_lib.CollectiveCommunication.NCCL) and
        self._num_gpus_per_worker == 0):
      logging.warning("Enabled NCCL communication but no GPUs detected/"
                      "specified.")

  def _in_multi_worker_mode(self):
    """Whether this strategy indicates working in multi-worker settings."""
    return self._num_workers > 1

  @property
  def experimental_between_graph(self):
    return True

  @property
  def experimental_should_init(self):
    return True

  @property
  def should_checkpoint(self):
    return self._is_chief

  @property
  def should_save_summary(self):
    return self._is_chief

  @property
  def _num_replicas_in_sync(self):
    return len(self.worker_devices) * self._num_workers

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True
