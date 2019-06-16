import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

f = open('../results/sacred/5/info.json')
ff = open('../results/sacred/10/info.json')
fff = open('../results/sacred/12/info.json')
ffff = open('../results/sacred/23/info.json')
coma_f = open('../results/sacred/14/info.json')

data = json.load(f)
data_2 = json.load(ff)
data_3 = json.load(fff)
data_4 = json.load(ffff)
data_coma = json.load(coma_f)
test_won_mean = data['test_battle_won_mean']
test_won_mean_t = data['test_battle_won_mean_T']
test_won_mean = np.array(test_won_mean).astype(np.float32)
test_won_mean_t = np.array(test_won_mean_t).astype(np.int)

test_won_mean_global_only = data_2['test_battle_won_mean']
test_won_mean_t_global_only  = data_2['test_battle_won_mean_T']
test_won_mean_global_only = np.array(test_won_mean_global_only).astype(np.float32)
test_won_mean_t_global_only  = np.array(test_won_mean_t_global_only ).astype(np.int)

test_won_mean_global_only_rnn = data_3['test_battle_won_mean']
test_won_mean_t_global_only_rnn = data_3['test_battle_won_mean_T']
test_won_mean_global_only_rnn = np.array(test_won_mean_global_only_rnn).astype(np.float32)
test_won_mean_t_global_only_rnn= np.array(test_won_mean_t_global_only_rnn).astype(np.int)

test_won_mean_global_only_rnn_train_more = data_4['test_battle_won_mean']
test_won_mean_t_global_only_rnn_train_more = data_4['test_battle_won_mean_T']
test_won_mean_global_only_rnn_train_more = np.array(test_won_mean_global_only_rnn_train_more).astype(np.float32)
test_won_mean_t_global_only_rnn_train_more= np.array(test_won_mean_t_global_only_rnn_train_more).astype(np.int)

test_won_mean_coma = data_coma['test_battle_won_mean']
test_won_mean_t_coma = data_coma['test_battle_won_mean_T']
test_won_mean_coma = np.array(test_won_mean_coma).astype(np.float32)
test_won_mean_t_coma= np.array(test_won_mean_t_coma).astype(np.int)

plt.figure(figsize=(15,4))
plt.grid()
sns.lineplot(test_won_mean_t, test_won_mean, label='qmix', lw=1.5)
sns.lineplot(test_won_mean_t_global_only, test_won_mean_global_only, label='global q only + mlp', lw=1.5)
# plt.plot(test_won_mean_t_global_only_rnn, test_won_mean_global_only_rnn, label='global q only + rnn', lw=1.5)
sns.lineplot(test_won_mean_t_global_only_rnn_train_more, test_won_mean_global_only_rnn_train_more, label='global q only + rnn', lw=1.5)
sns.lineplot(test_won_mean_t_coma, test_won_mean_coma, label='COMA', lw=1.5)

plt.legend()
plt.xlabel('T')
plt.ylabel('battle_won_mean')
plt.show()
