import copy
from components.episode_buffer import EpisodeBatch
from modules.potential.globalq import GlobalQ
from modules.agents import REGISTRY as agent_REGISTRY
import torch as th
from torch.optim import RMSprop



class PotentialQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        input_shape = self._get_input_shape(scheme)
        self.localQ = agent_REGISTRY["rnn"](input_shape, args)
        self.target_localQ = copy.deepcopy(self.localQ)
        self.logger = logger

        self.last_target_update_episode = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.globalQ = mac.agent
        self.target_globalQ = copy.deepcopy(self.globalQ)

        self.localQ_params = list(self.localQ.parameters())
        self.globalQ_params = list(self.globalQ.parameters())
        self.params = self.localQ_params + self.globalQ_params

        self.localQ_optimizer = RMSprop(params=self.localQ_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.globalQ_optimizer = RMSprop(params=self.globalQ_params, lr=args.global_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Optimize Global Q
        global_q_out = []
        hidden_states = self.globalQ.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
        for t in range(max_t):
            global_q, hidden_states = self.globalQ(batch, hidden_states, t=t)
            global_q_out.append(global_q.squeeze(1))
        global_q_out = th.stack(global_q_out, dim=1)
        chosen_g_action_qvals = th.gather(global_q_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        if th.cuda.is_available():
            default_actions = th.ones(actions.size(), dtype=th.long).cuda()
        else:
            default_actions = th.ones(actions.size(), dtype=th.long)
        default_g_action_qvals = th.gather(global_q_out[:, :-1], dim=3, index=default_actions).squeeze(3)

        target_global_q_out = []
        target_hidden_states = self.globalQ.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)

        for t in range(max_t):
            target_global_q, target_hidden_states = self.target_globalQ(batch, target_hidden_states, t=t)
            target_global_q_out.append(target_global_q.squeeze(1))
        target_global_q_out = th.stack(target_global_q_out[1:], dim=1)

        target_global_q_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            global_q_out[avail_actions == 0] = -9999999
            cur_max_actions = global_q_out[:, 1:].max(dim=3, keepdim=True)[1]
            target_g_max_qvals = th.gather(target_global_q_out, dim=3, index=cur_max_actions).squeeze(3)
        else:
            target_g_max_qvals = th.gather(target_global_q_out, dim=3, index=actions).squeeze(3)

        # Calculate 1-step Q-Learning targets
        targets_g = rewards + self.args.gamma * (1 - terminated) * target_g_max_qvals

        # Td-error
        td_error_g = (chosen_g_action_qvals - targets_g.detach())

        mask = mask.expand_as(td_error_g)

        # 0-out the targets that came from padded data
        masked_td_error_g = td_error_g * mask

        # Normal L2 loss, take mean over actual data
        loss_g = (masked_td_error_g ** 2).sum() / mask.sum()
        self.globalQ_optimizer.zero_grad()
        loss_g.backward()
        grad_norm_g = th.nn.utils.clip_grad_norm_(self.globalQ_params, self.args.grad_norm_clip)
        self.globalQ_optimizer.step()

        # for each local Q function
        # Calculate estimated Q-Values
        local_out = []
        hidden_states = self.localQ.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
        for t in range(max_t):
            agent_outs, hidden_states = self.localQ.forward(batch, hidden_states, t=t)
            local_out.append(agent_outs)
        local_out = th.stack(local_out, dim=1)  # Concat over time batch_size * seq_length * n_agents * n_actions

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(local_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        target_local_out = []
        target_hidden_states = self.target_localQ.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
        for t in range(max_t):
            target_agent_outs, target_hidden_states = self.target_localQ.forward(batch, target_hidden_states, t=t)
            target_local_out.append(target_agent_outs)
        target_local_out = th.stack(target_local_out[1:], dim=1)

        # Mask out unavailable actions
        target_local_out[avail_actions[:, 1:] == 0] = -9999999


        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            local_out[avail_actions == 0] = -9999999
            cur_max_actions = local_out[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_local_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_local_out.max(dim=3)[0]

        diff_rewards = chosen_g_action_qvals - default_g_action_qvals

        # Calculate 1-step Q-Learning targets
        targets = diff_rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        noop_mask = th.zeros(td_error.size())
        noop_index = (actions == 0).squeeze(3).byte()
        if th.cuda.is_available():
            noop_mask = noop_mask.cuda()
            noop_index = noop_index.cuda()

        noop_mask[~noop_index] = 1

        mask = mask.expand_as(td_error)
        noop_mask = noop_mask * mask

        # 0-out the targets that came from padded data
        masked_td_error = td_error * noop_mask #mask


        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / noop_mask.sum()
        if loss > 1e6:
            print('-----')
            print(((td_error * mask) ** 2).sum())
            print((masked_td_error ** 2).sum())
            print(mask.size(), noop_mask.size())
            for idd in range(len(noop_mask)):
                print(mask[idd] - noop_mask[idd])
            print(mask.sum(), noop_mask.sum(), loss.item())

        # Optimise
        self.localQ_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.localQ_params, self.args.grad_norm_clip)
        self.localQ_optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat('global_loss', loss_g.item(), t_env)
            self.logger.log_stat('global_grad_norm', grad_norm_g, t_env)
            mask_elems = mask.sum().item()
            noop_mask_elems = noop_mask.sum().item()
            self.logger.log_stat('global_td_error_abs', (masked_td_error_g.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat('global_q_taken_mean', (chosen_g_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat('global_target_mean', (targets_g * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat('default_g_action_qvals', (default_g_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat('diff_rewards', (diff_rewards * noop_mask).sum().item()/(noop_mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/noop_mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


    def _update_targets(self):
        self.target_localQ.load_state(self.localQ)
        self.target_globalQ.load_state_dict(self.globalQ.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.localQ.cuda()
        self.target_localQ.cuda()
        self.globalQ.cuda()
        self.target_globalQ.cuda()

    def save_models(self, path):
        th.save(self.localQ.state_dict(), "{}/agent.th".format(path))
        th.save(self.globalQ.state_dict(), "{}/critic.th".format(path))
        th.save(self.localQ_optimizer.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.globalQ_optimizer.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.localQ.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.globalQ.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_globalQ.load_state_dict(self.globalQ.state_dict())
        self.target_localQ.load_state(self.localQ)
        self.localQ_optimizer.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.globalQ_optimizer.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape