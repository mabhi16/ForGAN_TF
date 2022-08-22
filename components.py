from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Concatenate, LSTM, GRU, Reshape, ReLU


def build_Gen(noise_size, f_size, p_step, condition_size, generator_latent_size, cell_type):
    cond_inp = Input(shape=(condition_size, f_size,))
    noise_inp = Input(shape=(noise_size,))
    # norm_cond = Lambda(lambda x: ((x - mean) / std))(cond_inp)
    # rs_cond = Reshape((condition_size, 1))(cond_inp)
    if cell_type == 'lstm':
        rnn = LSTM(generator_latent_size)(cond_inp)
    else:
        rnn = GRU(generator_latent_size)(cond_inp)
    c_inp = Concatenate(axis=1)([rnn, noise_inp])
    lyr = Dense((generator_latent_size + noise_size), )(c_inp)
    lyr = ReLU()(lyr)
    oup = Dense(p_step*f_size)(lyr)
    rs_op = Reshape((p_step, f_size), )(oup)
    # denorm_op = Lambda(lambda x: (x * std + mean))(oup)
    Gen = Model([noise_inp, cond_inp], rs_op)
    return Gen


def build_Disc(condition_size, f_size, p_step, discriminator_latent_size, cell_type):
    cond_inp = Input(shape=(condition_size, f_size,))
    pred_inp = Input(shape=(p_step, f_size,))
    c_inp = Concatenate(axis=1)([cond_inp, pred_inp])
    # norm_inp = Lambda(lambda x: ((x - mean) / std))(c_inp)
    # rs_inp = Reshape((condition_size + 1, 1))(c_inp)
    if cell_type == 'lstm':
        rnn = LSTM(discriminator_latent_size)(c_inp)
    else:
        rnn = GRU(discriminator_latent_size)(c_inp)
    output = Dense((p_step*cond_inp.shape[2]), activation='sigmoid')(rnn)
    rs_op = Reshape((p_step, cond_inp.shape[2]),)(output)
    Disc = Model([pred_inp, cond_inp], rs_op)
    Disc.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return Disc


def comgan(gen, disc):
    disc.trainable = False
    noise, cond = gen.input
    fake_data = gen.output
    # fake_data = Reshape((1, 5))(fake_data)
    d_g_decision = disc([fake_data, cond])
    gan = Model([noise, cond], d_g_decision)
    # gan.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return gan
