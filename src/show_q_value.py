
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd



num_agents = 5
num_actions = 11
actions = [2, 3, 4, 5, 6, 7, 8, 9, 10]
actions_name = ('no-op', 'stop', 'move_north', 'move_south', 'move_east', 'move_west', 'attack0', 'attack1', 'attack2', 'attack3', 'attack4') # 1: 'stop'
save_path = '../results/models/potential_smac__2019-06-18_16-04-59'
save_dirs = os.listdir(save_path)
global_q_out = []
action_out = []
for dir in save_dirs:
    if int(dir) < 2000:
        data = np.load(os.path.join(save_path, dir, 'g_out.npy'))
        action_data = np.load(os.path.join(save_path, dir, 'a_out.npy'))
        for aa in action_data:
            length = aa.shape[1]
            aa = np.reshape(aa, (-1, num_agents))
            action_out.extend(aa)
        for d in data:
            d = np.reshape(d[:, :length, :, :], (-1, num_agents, num_actions))
            global_q_out.extend(d)


global_q_out = np.stack(np.array(global_q_out), axis=0)
action_out = np.stack(np.array(action_out), axis=0)

for n in range(num_agents):
    # plt.figure(figsize=(13, 5))
    # plt.title('The distribution of selected actions')
    # sns.distplot(action_out[:, n], kde=False)
    # plt.xticks(np.arange(11), actions_name)
    # plt.grid()
    # plt.xlabel('Actions')
    # plt.ylabel('# of selected actions')
    # plt.savefig('../../action_agent_{}_all.png'.format(n))

    a_sample = global_q_out[:, n, :]
    q_data = pd.DataFrame(columns=['actions', 'q'])
    for i in range(a_sample.shape[0]):
        q_data.append({"actions": 'default action', "q": global_q_out[i, n, 1]}, ignore_index=True)
        # for a in actions:
            # q_data = q_data.append({"actions": actions_name[a], "q": a_sample[i, a] - global_q_out[i, n, 1]}, ignore_index=True)
            # q_data = q_data.append({"actions": actions_name[a], "q": a_sample[i, a]}, ignore_index=True)
    plt.figure(figsize=(13, 9))
    plt.title('Global Q(a) : Agent ' + str(n))

    # plt.title('Global Q(a) - Global Q(default action): Agent ' + str(n))
    sns.boxplot(x='actions', y='q', data=q_data)
    plt.grid()
    plt.axhline(0, color='k')
    # plt.show()
    plt.savefig('../../q_agent_{}_only_default.png'.format(n))


print(global_q_out.shape[0])

