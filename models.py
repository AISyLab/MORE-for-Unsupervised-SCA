import tensorflow as tf
from keras.optimizers import *
from keras.layers import *
from keras.regularizers import *
from keras import *
import keras.backend as K


def pearson_correlation(y_true, y_pred):
    x = K.cast(y_true, dtype="float32")
    y = K.cast(y_pred, dtype="float32")
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.sqrt(tf.reduce_mean(tf.square(xm))) * tf.math.sqrt(tf.reduce_mean(tf.square(ym)))
    r = r_num / r_den
    r = tf.where(tf.math.is_nan(r), 0., r)
    return 1 - tf.abs(r)


def z_score_pearson_correlation(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    mx = tf.reduce_mean(z_true)
    my = tf.reduce_mean(z_pred)
    xm, ym = z_true - mx, z_pred - my
    r_num = tf.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.sqrt(tf.reduce_mean(tf.square(xm))) * tf.math.sqrt(tf.reduce_mean(tf.square(ym)))
    r = r_num / r_den
    r = tf.where(tf.math.is_nan(r), 0., r)
    return 1 - tf.abs(r)


def z_score_mse(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    return tf.reduce_mean(tf.math.square(z_true - z_pred))


def z_score_mae(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    return tf.reduce_mean(tf.math.abs(z_true - z_pred))


def tukeys_biweight(y_true, y_pred):
    delta = 4.5
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    error = y_true - y_pred
    abs_error = tf.abs(error)

    error_sq = (y_true - y_pred) ** 2
    mask_below = tf.cast((abs_error <= delta), tf.float32)
    rho_above = tf.cast((abs_error > delta), tf.float32) * delta / 2

    rho_below = (delta / 2) * (1 - ((1 - ((error_sq * mask_below) / delta)) ** 3))
    rho = rho_above + rho_below

    return tf.reduce_mean(rho)


def log_cosh(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    error = y_true - y_pred
    abs_error = tf.math.log(tf.cosh(error))
    loss = abs_error
    return tf.reduce_sum(loss)


def z_score_log_cosh(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    error = z_true - z_pred
    abs_error = tf.math.log(tf.cosh(error))
    loss = abs_error
    return tf.reduce_sum(loss)


def huber(y_true, y_pred):
    delta = 1.0
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(loss)


def z_score_huber(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    delta = 1
    error = z_true - z_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(loss)


def get_loss(loss):
    loss_dict = {
        "mse": "mse",
        "mae": "mae",
        "corr": pearson_correlation,
        "z_score_corr": z_score_pearson_correlation,
        "z_score_mse": z_score_mse,
        "z_score_mae": z_score_mae,
        "z_score_huber": z_score_huber,
        "huber": tf.keras.losses.Huber(delta=1),
        "logcosh": tf.keras.losses.LogCosh,
        "z_score_log_cosh": z_score_log_cosh,
        "tukeys_biweight": tukeys_biweight

    }
    return loss_dict[loss]


def get_optimizer(optimizer, lr):
    if optimizer == "Adam":
        return Adam(lr=lr)
    else:
        return RMSprop(lr=lr)


def cnn(number_of_samples, loss, hp):
    tf.random.set_seed(hp["seed"])

    input_shape = (number_of_samples, 1)
    input_layer = Input(shape=input_shape, name="input_layer")

    if hp["kernel_regularizer"] == "l1":
        regularizer = l1(l=hp["kernel_regularizer_value"])
    elif hp["kernel_regularizer"] == "l2":
        regularizer = l2(l=hp["kernel_regularizer_value"])
    else:
        regularizer = None

    x = None
    x = Conv1D(hp['filters'], hp['kernel_size'], strides=hp['strides'], activation=hp['activation'],
               kernel_initializer=hp['kernel_initializer'], kernel_regularizer=regularizer, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = AveragePooling1D(hp['pool_size'], strides=hp['pool_strides'], padding='same')(x)
    for l_i in range(1, hp["conv_layers"]):
        x = Conv1D(hp['filters'] * (l_i + 1), hp['kernel_size'], strides=hp['strides'], activation=hp['activation'],
                   kernel_initializer=hp['kernel_initializer'], kernel_regularizer=regularizer, padding='same')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(hp['pool_size'], strides=hp['pool_strides'], padding='same')(x)
    x = Flatten()(x)

    x = Dense(hp['neurons'], activation=hp['activation'], kernel_initializer=hp['kernel_initializer'],
              kernel_regularizer=regularizer, name=f'fc_1')(x)
    for l_i in range(1, hp["layers"]):
        x = Dense(hp['neurons'], activation=hp['activation'], kernel_initializer=hp['kernel_initializer'],
                  kernel_regularizer=regularizer, name=f'fc_{l_i + 1}')(x)
        if hp["dropout"] > 0:
            x = Dropout(rate=hp["dropout"])(x)

    output_layer = Dense(256, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='cnn_search')
    m_model.compile(loss=get_loss(loss), optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]))
    m_model.summary()
    return m_model


def mlp(number_of_samples, loss, hp):
    tf.random.set_seed(hp["seed"])

    input_shape = (number_of_samples)
    input_layer = Input(shape=input_shape, name="input_layer")

    if hp["kernel_regularizer"] == "l1":
        regularizer = l1(l=hp["kernel_regularizer_value"])
    elif hp["kernel_regularizer"] == "l2":
        regularizer = l2(l=hp["kernel_regularizer_value"])
    else:
        regularizer = None

    x = None
    for l_i in range(hp["layers"]):
        x = Dense(hp["neurons"],
                  activation=hp["activation"],
                  kernel_initializer=hp["kernel_initializer"],
                  kernel_regularizer=regularizer,
                  name=f"layer_{l_i}")(
            input_layer if l_i == 0 else x)
        if hp["dropout"] > 0:
            x = Dropout(rate=hp["dropout"])(x)
    output = Dense(256, activation="linear", name='scores')(x)

    m_model = Model(input_layer, output, name='mlp_search')
    m_model.compile(loss=get_loss(loss), optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]))
    m_model.summary()
    return m_model
