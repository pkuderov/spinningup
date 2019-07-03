from typing import Tuple

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context


def cyclic_learning_rate(
        global_step,
        step_size: int,
        learning_rate: Tuple[float, float],
        const_lr_decay=.5,
        max_lr_decay=.7,
        name=None
):
    """Applies cyclic learning rate (CLR).
       From the paper:
       Smith, Leslie N. "Cyclical learning
       rates for training neural networks." 2017.
       [https://arxiv.org/pdf/1506.01186.pdf]
        This method lets the learning rate cyclically
       vary between reasonable boundary values
       achieving improved classification accuracy and
       often in fewer iterations.
        This code varies the learning rate linearly between the
       minimum (learning_rate) and the maximum (max_lr).
        It returns the cyclic learning rate. It is computed as:
         ```python
         cycle = floor( 1 + global_step /
          ( 2 * step_size ) )
        x = abs( global_step / step_size – 2 * cycle + 1 )
        clr = learning_rate +
          ( max_lr – learning_rate ) * max( 0 , 1 - x )
         ```
        Polices:
            The learning rate varies between the minimum and maximum
            boundaries and each boundary value declines by an exponential
            factor of: gamma^global_step.
         Example:
          '''python
          ...
          global_step = tf.Variable(0, trainable=False)
          optimizer = tf.train.AdamOptimizer(learning_rate=
            clr.cyclic_learning_rate(global_step=global_step, step_size=8))
          train_op = optimizer.minimize(loss_op, global_step=global_step)
          ...
           with tf.Session() as sess:
              sess.run(init)
              for step in range(1, num_steps+1):
                assign_op = global_step.assign(step)
                sess.run(assign_op)
          ...
           '''
         Args:
          global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
            Global step to use for the cyclic computation.  Must not be negative.
          step_size: A scalar. The number of iterations in half a cycle.
            The paper suggests step_size = 2-8 x training iterations in epoch.
          learning_rate: A tuple of two scalars (min_lr, max_lr), where every scalar is `float32` or
          `float64` `Tensor` or a Python number.
            min_lr: The initial learning rate which is the lower bound of the cycle. Never changes.
                Actual learning rate tends to it.
            max_lr: The maximum learning rate boundary.
          const_lr_decay: float constant. Defines the base for the decay, const_lr_decay ** cycle,
            which's applied to (max_lr - min_lr) and defines the const part of learning rate during the cycle.
          max_lr_decay: float constant. Defines the base for the decay of learning rate' dynamic part,
            max_lr_decay ** cycle.
          name: String.  Optional name of the operation.  Defaults to
            'CyclicLearningRate'.
         Returns:
          A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
          learning rate.
        Raises:
          ValueError: if `global_step` is not supplied.
        @compatibility(eager)
        When eager execution is enabled, this function returns
        a function which in turn returns the decayed learning
        rate Tensor. This can be useful for changing the learning
        rate value across different invocations of optimizer functions.
        @end_compatibility
    """
    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")

    min_lr, max_lr = learning_rate
    with ops.name_scope(name, "CyclicLearningRate", [global_step]) as name:
        step_size = float(step_size)
        global_step = tf.cast(global_step, tf.float32)

        def cyclic_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2., step_size)
            global_div_double_step = math_ops.divide(global_step, double_step)
            cycle_from_zero = math_ops.floor(global_div_double_step)
            cycle = math_ops.add(1., cycle_from_zero)
            # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
            double_cycle = math_ops.multiply(2., cycle)
            global_div_step = math_ops.divide(global_step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1., tmp))

            # compute const part with decay
            # const_lr = min_rl + (max_lr - min_lr) * const_lr_decay ** cycle
            const_lr_diff = math_ops.multiply(
                math_ops.subtract(max_lr, min_lr),
                math_ops.pow(const_lr_decay, cycle)
            )
            const_lr = math_ops.add(min_lr, const_lr_diff)

            # compute dynamic part with decay
            dyn_lr = math_ops.multiply(
                math_ops.subtract(max_lr, const_lr),
                math_ops.maximum(0., math_ops.subtract(1., x))
            )
            dynamic_decay = math_ops.pow(max_lr_decay, cycle_from_zero)
            dyn_lr = math_ops.multiply(dyn_lr, dynamic_decay)

            # compute learning rate
            # learning_rate = const_lr + dyn_lr
            return math_ops.add(const_lr, dyn_lr, name=name)

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()
        return cyclic_lr
