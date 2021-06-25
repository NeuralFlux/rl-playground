import random

from const import POS_NUMBERS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PlayerNet(nn.Module):

    def __init__(self, **kwargs):
        super(PlayerNet, self).__init__()

        # hidden layers
        self.fc1 = nn.Linear(
            kwargs["input_size"],
            kwargs["hidden_layer_size"]
        )
        self.fc2 = nn.Linear(
            kwargs["hidden_layer_size"],
            kwargs["hidden_layer_size"]
        )

        # output layers
        self.hidden_head = nn.Linear(
            kwargs["hidden_layer_size"],
            kwargs["hidden_state_size"]
        )
        self.q_head = nn.Linear(
            kwargs["hidden_layer_size"],
            kwargs["output_size"]
        )

    def forward(self, x):
        """
        x: [Input] + [Hidden state]
        returns: [Q values], [Next hidden state]
        """

        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        hidden_out = self.hidden_head(h2)  # NOTE ReLU?
        q_values = self.q_head(h2)  # NOTE Okay?

        return (q_values, hidden_out)


class NNPlayer(object):

    def __init__(self) -> None:

        # none bit + 648 actions one-hot + [bulls, cows] + hidden
        input_size = 1 + 648 + 2 + 64
        self.net = PlayerNet(
            input_size=input_size,
            hidden_layer_size=64,
            hidden_state_size=64,
            output_size=648
        )

        # personal replay buffer
        self.p_rep_buffer = {
            'states': [],
            'actions': [],
            'rewards': []
        }

        self.first_act = False
        self.none_bit = torch.ones(1, dtype=torch.bool)
        self.last_guess = torch.zeros(648, dtype=torch.int)
        self.score = torch.zeros(2, dtype=torch.int)
        self.hidden_state = torch.zeros(64, dtype=torch.float)
    
    def get_secret_num(self, first_player):
        self.first_player = first_player
        return random.choice(POS_NUMBERS)
    
    def get_guess(self):
        # get the best guess
        best_act_idx = self._get_optimal_action().item()
        best_guess = POS_NUMBERS[best_act_idx]

        # only used for first action since last_guess is none
        if not self.first_act:
            self.none_bit = torch.zeros(1, dtype=torch.bool)
            self.first_act = True

        # remember last guess
        self.last_guess = F.one_hot(
            torch.tensor( best_act_idx, dtype=torch.int64 ),
            648
        )

        return best_guess

    def _get_optimal_action(self):
        # TODO check if proper concat happens
        net_input = torch.cat((
            self.none_bit,
            self.last_guess,
            self.score,
            self.hidden_state
        ), dtype=torch.float)

        q_vals, curr_hidden_state = self.net.forward(net_input)
        self.hidden_state = curr_hidden_state

        action = torch.argmax(q_vals)

        self.p_rep_buffer['states'].append(net_input)  # NOTE python bug?
        self.p_rep_buffer['actions'].append(action)
        return action

    def set_score(self, score):
        self.score = torch.tensor(score, dtype=torch.float)

        reward = torch.dot(self.score, torch.tensor([1, 0.5]))
        reward -= 0.1  # penalise every step
        self.p_rep_buffer['rewards'].append(reward)

    def finish_game(self, summary):
        result, _ = summary
        win, draw = False, (result == 0)
        if self.first_player:
            if result == 1:
                win = True
        else:
            if result == -1:
                win = True
        
        # change last reward
        self.p_rep_buffer['rewards'][-1] = torch.tensor(
            (win * 7) + (draw * -4) + ((not win and not draw) * -7),
            dtype=torch.float
        )
