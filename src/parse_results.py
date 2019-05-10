import json
import numpy as np
import matplotlib.pyplot as plt

f = open('../results/sacred/6/info.json')
data = json.load(f)
test_won_mean = data['default_g_action_qvals']
test_won_mean_t = data['default_g_action_qvals_T']
test_won_mean = np.array(test_won_mean).astype(np.float32)
test_won_mean_t = np.array(test_won_mean_t).astype(np.int)
plt.figure(figsize=(10,3))
plt.grid()
plt.plot(test_won_mean_t, test_won_mean)
plt.xlabel('T')
plt.ylabel('default_g_action_qvals')
plt.show()
