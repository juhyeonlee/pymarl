import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        self.hyper_w1_wlu = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w2_wlu = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b1_wlu = nn.Linear(self.state_dim, self.embed_dim)

        self.V_wlu = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        ##TODO: onehot 만들어서 각 에이전트마다 씌우기
        # output dimension은 batch size * time steps * n_agents
        # q_tot - q(-u) = u agent의 q 값
        # loss function 생각 해보기

        self.wlu_mask = th.eye(self.n_agents, device=args.device)


    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents) # (batch size * time steps) * 1 * n_agent (max q or selected q)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1) # batch size * step size * 1

        agent_qs_wlu_masked = self.wlu_mask.unsqueeze(0) * agent_qs.expand(-1, self.n_agents, -1) # bs * n_agent * n_agent
        agent_qs_wlu_masked = agent_qs_wlu_masked.view(-1, 1, self.n_agents)

        # w1_wlu = th.abs(self.hyper_w1_wlu(states))
        # b1_wlu = self.hyper_b1_wlu(states)
        # w1_wlu = w1_wlu.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.n_agents, self.embed_dim)
        # b1_wlu = b1_wlu.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, 1, self.embed_dim)
        #
        # hidden_wlu = F.elu(th.bmm(agent_qs_wlu_masked, w1_wlu) + b1_wlu)
        #
        # w2_wlu = th.abs(self.hyper_w2_wlu(states))
        # w2_wlu = w2_wlu.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.embed_dim, 1)
        # v_wlu = self.V_wlu(states).unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, 1, 1)
        #
        # y_wlu = th.bmm(hidden_wlu, w2_wlu) + v_wlu
        #
        # q_wlu = y_wlu.view(bs, -1, self.n_agents)
        hidden_wlu = F.elu(th.bmm(agent_qs_wlu_masked, w1) + b1)
        y_wlu = th.bmm(hidden_wlu, w_final) + v
        q_wlu = y_wlu.view(bs, -1, self.n_agents)

        return q_tot, q_wlu
