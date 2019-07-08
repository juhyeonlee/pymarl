import json
import numpy as np
import matplotlib.pyplot as plt

f = open('../results/sacred/31/info.json')
data = json.load(f)

global_q_taken_mean = data['default_g_action_qvals']
global_q_taken_mean_t = data['default_g_action_qvals_T']
global_q_taken_mean = np.array(global_q_taken_mean).astype(np.float32)
global_q_taken_mean_t = np.array(global_q_taken_mean_t).astype(np.int)

# test_won_mean_global_only = data_2['test_battle_won_mean']
# test_won_mean_t_global_only  = data_2['test_battle_won_mean_T']
# test_won_mean_global_only = np.array(test_won_mean_global_only).astype(np.float32)
# test_won_mean_t_global_only  = np.array(test_won_mean_t_global_only ).astype(np.int)
#
# test_won_mean_global_only_rnn = data_3['test_battle_won_mean']
# test_won_mean_t_global_only_rnn = data_3['test_battle_won_mean_T']
# test_won_mean_global_only_rnn = np.array(test_won_mean_global_only_rnn).astype(np.float32)
# test_won_mean_t_global_only_rnn= np.array(test_won_mean_t_global_only_rnn).astype(np.int)
plt.figure(figsize=(11,4))
plt.grid()
plt.plot(global_q_taken_mean_t, global_q_taken_mean, label='default_g_action_qvals', lw=1.5)
# plt.plot(test_won_mean_t_global_only, test_won_mean_global_only, label='global q only + mlp', lw=1.5)
# plt.plot(test_won_mean_t_global_only_rnn, test_won_mean_global_only_rnn, label='global q only + rnn', lw=1.5)

plt.legend()
plt.xlabel('T')
plt.ylabel('default_g_action_qvals')
plt.show()