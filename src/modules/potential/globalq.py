import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GlobalQ(nn.Module):
    def __init__(self, scheme, args):
        super(GlobalQ, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.rnn_hidden_dim = 128

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        # self.fc2 = nn.Linear(128, 128)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc3 = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def forward(self, batch, hidden_state, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        # print(x.size(), hidden_state.size())
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        x_in = x.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x_in, h_in)
        q = self.fc3(h)
        q = q.reshape(batch.batch_size, self.n_agents, -1)
        h = h.reshape(batch.batch_size, self.n_agents, -1)
        return q, h

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape