import json
import numpy as np
import matplotlib.pyplot as plt

f = open('../results/sacred/5/info.json')
ff = open('../results/sacred/10/info.json')
data = json.load(f)
data_2 = json.load(ff)
test_won_mean = data['test_battle_won_mean']
test_won_mean_t = data['test_battle_won_mean_T']
test_won_mean = np.array(test_won_mean).astype(np.float32)
test_won_mean_t = np.array(test_won_mean_t).astype(np.int)

test_won_mean_global_only = data_2['global_q_taken_mean']
test_won_mean_t_global_only  = data_2['global_q_taken_mean_T']
test_won_mean_global_only = np.array(test_won_mean_global_only).astype(np.float32)
test_won_mean_t_global_only  = np.array(test_won_mean_t_global_only ).astype(np.int)
plt.figure(figsize=(10,3))
plt.grid()
# plt.plot(test_won_mean_t, test_won_mean, label='qmix')
plt.plot(test_won_mean_t_global_only, test_won_mean_global_only, label='global q only')
plt.legend()
plt.xlabel('T')
plt.ylabel('global_q_taken_mean')
plt.show()
