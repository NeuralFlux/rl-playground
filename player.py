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

        self.first_act = False
        self.none_bit = torch.ones(1, dtype=torch.bool)
        self.last_guess = torch.zeros(648, dtype=torch.int)
        self.score = torch.zeros(2, dtype=torch.int)
        self.hidden_state = torch.zeros(64, dtype=torch.float)
    
    def get_secret_num(self):
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
        ))

        q_vals, curr_hidden_state = self.net.forward(net_input)
        self.hidden_state = curr_hidden_state

        return torch.argmax(q_vals)
        
    
    def set_score(self, score):
        self.score = torch.int(score)

    def finish_game(self, summary):
        # result, turns = summary
        return
