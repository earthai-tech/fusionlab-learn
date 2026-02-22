import tensorflow as tf

from fusionlab.nn.components.heads import (
    GaussianHead,
    QuantileHead,
    MixtureDensityHead,
)

@tf.function
def run_all(x2, x3):
    gh = GaussianHead(output_dim=2)
    qh = QuantileHead([0.1, 0.5, 0.9], output_dim=2)
    mh = MixtureDensityHead(output_dim=2, num_components=5)

    g2 = gh(x2)
    g3 = gh(x3)

    q2 = qh(x2)
    q3 = qh(x3)

    m2 = mh(x2)
    m3 = mh(x3)

    # weights sum to 1 over K axis (-2 in your layout)
    tf.debugging.assert_near(
        tf.reduce_sum(m3["weights"], axis=-2),
        tf.ones_like(m3["weights"][..., 0, :]),
        atol=1e-5,
    )
    return g2, g3, q2, q3, m2, m3

x2 = tf.random.normal((4, 16))
x3 = tf.random.normal((4, 7, 16))
run_all(x2, x3)
print("OK")
