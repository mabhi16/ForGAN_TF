import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import tensorflow as tf
from utils import calc_kld
from components import build_Gen, build_Disc, comgan

df_ge = pd.read_csv('data_stock.csv', engine='python')
print("checking if any null values are present\n", df_ge.isna().sum())
train_cols = ["Open", "High", "Low", "Close", "Volume"]
df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
df_train, df_val = train_test_split(df_train, train_size=0.9, test_size=0.1, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
min_max_scaler = MinMaxScaler()
x_tr = df_train.loc[:, train_cols].values
x_tr = min_max_scaler.fit_transform(x_tr)
x_va = min_max_scaler.fit_transform(df_val.loc[:, train_cols].values)
x_te = min_max_scaler.fit_transform(df_test.loc[:, train_cols].values)
TIME_STEPS = 8


def build_timeseries(mat, ycol):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros(dim_0)

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, [ycol]]
    print("length of time-series i/o", x.shape, y.shape)
    return x, y


x_tr, y_tr = build_timeseries(x_tr, 3)
x_va, y_va = build_timeseries(x_va, 3)
x_te, y_te = build_timeseries(x_te, 3)
# disc_seq = x_tr[:, :, [3]]
# y_tr = np.reshape(y_tr, [y_tr.shape[0], 1, 5])
d_iter = 2
n_steps = 5000
batch_size = 1000
noise_size = 32
gen_lat_size = 8
cond_size = 8
disc_lat_size = 16
hist_bins = 80
hist_min = -2
hist_max = 2
rs = np.random.RandomState(1368)
generator = build_Gen(noise_size, x_tr.shape[2], 1, cond_size, gen_lat_size, 'gru')
discriminator = build_Disc(cond_size, x_tr.shape[2], 1, disc_lat_size, 'gru')
cgan = comgan(generator, discriminator)

print("\nNetwork Architecture\n")
print(generator.summary())
print("\n************************\n")
print(discriminator.summary())
print("\n************************\n")
cgan = comgan(generator, discriminator)
cgan.compile(loss='binary_crossentropy', optimizer='rmsprop')
print(cgan.summary())
print("\n************************\n")
x_tr = np.cast['float32'](x_tr)
y_tr = np.cast['float32'](y_tr)
x_va = np.cast['float32'](x_va)
y_va = np.cast['float32'](y_va)
best_kld = np.inf

for step in range(n_steps):
   d_loss = 0
   for _ in range(d_iter):
       # train discriminator on real data
       idx = rs.choice(x_tr.shape[0], batch_size)
       condition = x_tr[idx]
       real_data = y_tr[idx]
       noise_batch = rs.normal(0, 1, (condition.shape[0], noise_size))
       tar_real = tf.ones((condition.shape[0], 5), dtype=tf.float32)
       tar_fake = tf.zeros((condition.shape[0], 5), dtype=tf.float32)
       d_real_loss = discriminator.train_on_batch([real_data, condition], tar_real)
       d_loss += d_real_loss
       x_fake = generator.predict([noise_batch, condition])
       x_fake = np.reshape(x_fake, [batch_size, 1, x_fake.shape[1]])
       d_fake_loss = discriminator.train_on_batch([x_fake, condition], tar_fake)
       d_loss += d_fake_loss

   d_loss = d_loss / (2 * d_iter)
   noise_batch = rs.normal(0, 1, (batch_size, noise_size))
   noise_batch = np.cast['float32'](noise_batch)
   tar_gan = tf.ones((condition.shape[0], 5), dtype=tf.float32)
   g_loss = cgan.train_on_batch([noise_batch, condition], tar_gan)
   noise_batch = rs.normal(0, 1, (x_va.shape[0], noise_size))
   noise_batch = np.cast['float32'](noise_batch)
   preds = generator.predict([noise_batch, x_va])
   kld = calc_kld(preds, y_va, hist_bins, hist_min, hist_max)
   if kld <= best_kld and kld != np.inf:
       best_kld = kld
       print("step : {} , KLD : {}, RMSE : {}".format(step, best_kld,
                                                      np.sqrt(np.square(preds - y_va).mean())))
       generator.save("GE_multi_model")
       # self.generator.save_weights("./{}/best_tf.h5".format(self.opt.dataset), save_format='h5')

   if step % 100 == 0:
       print("step : {} , d_loss : {} , g_loss : {}".format(step, d_loss, g_loss))

print("Training Completed, working on testing")
rc_model = tf.keras.models.load_model("GE_model")
# y_test = y_te.flatten()
preds = []
rmses = []
maes = []
mapes = []
x_te = np.cast['float32'](x_te)
y_te = np.cast['float32'](y_te)

mean_pred = np.zeros(x_te.shape[0])
for _ in range(100):
    noise_batch = tf.convert_to_tensor(rs.normal(0, 1, (x_te.shape[0], noise_size)),
                                       dtype=tf.float32)
    pred = rc_model.predict([noise_batch, x_te])
    mean_pred = mean_pred + pred
    preds.append(pred)
    # pred.flatten()

    error = pred - y_te
    rmses.append(np.sqrt(np.square(error).mean()))
    maes.append(error.mean())
    # mapes.append((error / y_test).mean() * 100)
mean_pred = mean_pred / 100
# mean_pred.flatten()
preds = np.vstack(preds)
# preds = preds.flatten()
kld = calc_kld(preds, y_te, hist_bins, hist_min, hist_max)
print("Test resuts:\nRMSE : {}({})\nMAE : {}({})\nKLD : {}\n"
      .format(np.mean(rmses), np.std(rmses),
              np.mean(maes), np.std(maes),
              # np.mean(mapes), np.std(mapes),
              kld))
pred_open = mean_pred[:, [1]]
# pred_high = mean_pred[:, [1]]
# pred_low = mean_pred[:, [2]]
# pred_close = mean_pred[:, [3]]
# pred_vol = mean_pred[:, [4]]
gt_open = y_te
# gt_high = y_te[:, [1]]
# gt_low = y_te[:, [2]]
# gt_close = y_te[:, [3]]
# gt_vol = y_te[:, [4]]
# y_min = -0.1
# y_max = 1.2
# y_axis = range(0, 3999, 1)
# gru_pred = np.load('pred_gru.npy')
# gru_gt = np.load('gt_gru.npy')
# fig, ax = plt.subplots(2, 2)
# ax[0][0].plot(pred_open, color='blue', label='Predictions(Open)')
# ax[0][0].plot(gt_open, color='red', label='GT(open)')
# ax[0][0].set_title('Open')
# ax[0][0].legend()
# ax[0][1].plot(pred_high, color='blue', label='Predictions(High)')
# ax[0][1].plot(gt_high, color='red', label='GT(High)')
# ax[0][1].set_title('High')
# ax[0][1].legend()
# ax[0][2].plot(pred_low, color='blue', label='Predictions(Low)')
# ax[0][2].plot(gt_low, color='red', label='GT(Low)')
# ax[0][2].set_title('Low')
# ax[0][2].legend()
# ax[1][0].plot(pred_close, color='blue', label='Predictions(Close)')
# ax[1][0].plot(gt_close, color='red', label='GT(Close)')
# ax[1][0].set_title('Close')
# ax[1][1].legend()
# ax[1][1].plot(pred_vol, color='blue', label='Predictions(Close)')
# ax[1][1].plot(gt_vol, color='red', label='GT(Close)')
# ax[1][1].set_title('Volume')
# ax[1][1].legend()
plt.title('Predictions on test set(FORGAN)')
plt.xlabel('Days')
plt.ylabel('Scaled price')
plt.xlim(0, 2000)
plt.plot(pred_open, color='blue', label='Predictions(Open)')
plt.plot(gt_open, color='red', label='GT(open)')
plt.legend()
plt.show()
