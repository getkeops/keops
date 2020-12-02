"""
TensorFlow, with an XLA backend
====================================
"""


import tensorflow as tf

# tf.config.optimizer.set_jit(True)

from time import time

# Make sure that we're using the v2.0.0
print(tf.__version__)

# Our function, that XLA is going to compile
def KP(x, y, p):
    D_ij = tf.math.reduce_sum((x - y) ** 2, axis=2)
    K_ij = tf.math.exp(-D_ij)
    return K_ij @ p


nits = 100
Ns, D = [10000, 100000, 1000000], 3

#############################################
#
#

# First, test without XLA
for N in Ns[:1]:

    # Generate the data
    x = tf.random.normal((N, 1, D))
    y = tf.random.normal((1, N, D))
    p = tf.random.normal((N, 1))

    # First run just in case...
    out = KP(x, y, p)

    # Timings for TF vanilla
    start = time()
    for _ in range(nits):
        out = KP(x, y, p)

    # N.B.: we need some kind of "print" statement to make
    #       sure that TF actually executes our code
    print("TF Vanilla : ", out[:10])
    end = time()
    print("Timing with {} points: {} x {:.4f}s".format(N, nits, (end - start) / nits))

##############################################
#

# Second, test with XLA
for N in Ns:

    # Generate the data
    x = tf.random.normal((N, 1, D))
    y = tf.random.normal((1, N, D))
    p = tf.random.normal((N, 1))

    # Precompile just in case...
    out = tf.xla.experimental.compile(KP, inputs=[x, y, p])

    start = time()

    for _ in range(nits):
        out = tf.xla.experimental.compile(KP, inputs=[x, y, p])[0]

    # N.B.: we need some kind of "print" statement to make
    #       sure that TF actually executes our code
    print("XLA : ", out[:10])
    end = time()

    print("Timing with {} points: {} x {:.4f}s".format(N, nits, (end - start) / nits))
