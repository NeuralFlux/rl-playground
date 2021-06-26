"""
648 actions
K max turns per game
N exploratory games between best network and itself
10 * (2 * K * N) replay buffer
64 mini-batch
100 games between latest and best
25% lose rate at most of latest to become best (33% random)
detach gradient every 5th move to avoid vanishing
do not tell each player their chance (1st or 2nd)
"""

from collections import deque
from random import random

from const import GAMMA, POS_NUMBERS

from player import NNPlayer, PlayerNet

import torch
import torch.nn.functional as F
import torch.optim as optim


class Trainer(object):

    def __init__(self, max_rep_size) -> None:
        # (STATE, ACTION, REWARD, NEXT_STATE, IS_NEXT_STATE_TERMINAL)
        self.replay_buffer = deque(maxlen=max_rep_size)

        # none bit + 648 actions one-hot + [bulls, cows] + hidden
        input_size = 1 + 648 + 2 + 64
        self.best_nn = PlayerNet(
            input_size=input_size,
            hidden_layer_size=64,
            hidden_state_size=64,
            output_size=648
        )
        self.optimizer = optim.Adam(self.best_nn.parameters(), lr=0.0001)

        self.latest_nn = None
    
    def simulate(self, eps, num_matches, max_turns, style='exploratory'):
        if style == 'exploratory':
            feedforward_api_one = self.best_nn.forward
            feedforward_api_two = self.best_nn.forward
        elif style == 'competition':
            feedforward_api_one = self.best_nn.forward
            feedforward_api_two = self.latest_nn.forward

        # store results
        results = torch.zeros(num_matches, dtype=torch.int64)

        # play matches
        for m_idx in range(num_matches):
            player_one = NNPlayer(feedforward_api_one, exp_const=eps)
            player_two = NNPlayer(feedforward_api_two, exp_const=eps)

            game = Game()
            result, _ = game.start_game( (player_one, player_two), max_turns )
            results[m_idx] = result

            if style == 'exploratory':
                # NOTE destroyed if p_rep_buffer is deconstructed?
                self._extract_personal_buffer(player_one.p_rep_buffer)
                self._extract_personal_buffer(player_two.p_rep_buffer)
        
        if style == 'competition':
            return results

    def _extract_personal_buffer(self, buffer):
        for idx in range(len(buffer) - 1):
            self.replay_buffer.append( (
                buffer[idx][0],  # S
                buffer[idx][1],  # A
                buffer[idx][2],  # R
                buffer[idx + 1][0],  # S'
                buffer[idx][3],
            ) )
            assert not buffer[idx][3]

        # NOTE skip the last exp if its not terminal 
        # FIXME devise a better procedure to avoid that
        if buffer[-1][3]:
            # append last exp with S' as S
            self.replay_buffer.append( (
                buffer[-1][0],  # S
                buffer[-1][1],  # A
                buffer[-1][2],  # R
                buffer[-1][0],  # S'
                buffer[-1][3],
            ) )

    def train(self, mb_size):
        batch = random.sample(self.replay_buffer, mb_size)

        # pre-process batch
        states = torch.zeros((mb_size, 650 + 65), dtype=torch.float)
        actions = torch.zeros((mb_size), dtype=torch.int64)
        rewards = torch.zeros((mb_size), dtype=torch.float)
        next_states = torch.zeros((mb_size, 650 + 65), dtype=torch.float)
        is_terminal = torch.zeros((mb_size), dtype=torch.bool)

        for idx, transition in enumerate(batch):
            states[idx] = transition[0]
            actions[idx] = transition[1]
            rewards[idx] = transition[2]
            next_states[idx] = transition[3]
            is_terminal[idx] = transition[4]

        # get q(s, a)
        obs_q_vectors = self.best_nn.forward(states)
        obs_q_values = obs_q_vectors.gather(1, actions.unsqueeze(1))

        # get q'(s, a) from next_state
        obs_next_q_vectors = self.best_nn.forward(next_states)
        # v(s') = max_a q(s', a)
        estim_v_values = obs_next_q_vectors.max(1)[0].detach()
        expected_q_values = rewards + is_terminal * GAMMA * estim_v_values

        loss = F.smooth_l1_loss(obs_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.best_nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class Game(object):

    FULL_SCORE = (3, 0)
    def __init__(self) -> None:
        pass

    def start_game(self, players, max_turns):
        self.players = players
        self.NUM_PLAYERS = len(self.players)
        assert self.NUM_PLAYERS == 2

        # get secret numbers of each players
        self.secrets = [
            self.players[0].get_secret_num(first_player=True),
            self.players[1].get_secret_num(first_player=False)
        ]

        # validate secrets
        assert self.secrets[0] in POS_NUMBERS
        assert self.secrets[1] in POS_NUMBERS

        result = None
        turns = 0
        while (result is None and turns < max_turns):
            # play a round and compare scores
            scores = self._play_round()
            if scores[0] == Game.FULL_SCORE:
                result += 1
            if scores[1] == Game.FULL_SCORE:
                result -= 1

            turns += 1
            players[0].set_score(scores[0])
            players[1].set_score(scores[1])

        # if max_turns over, return draw
        if result is None:
            return (0, turns)

        # inform players of the result
        self.players[0].finish_game((result, turns))
        self.players[1].finish_game((result, turns))

        return (result, turns)

    def _play_round(self):
        guesses = [
            self.players[0].get_guess(),
            self.players[1].get_guess()
        ]

        # validate guesses
        assert guesses[0] in POS_NUMBERS
        assert guesses[1] in POS_NUMBERS

        scores = [
            self._score_guess(guesses[0], self.secrets[1]),
            self._score_guess(guesses[1], self.secrets[0])
        ]

        return scores

    def _score_guess(self, guess, secret):
        bulls, cows = 0, 0
        rem_idxs = set()
        rem_secret = ''

        str_guess = str(guess)
        str_secret = str(secret)

        # calculate bulls first and obtain remaining portion of string
        for idx in range(len(str_guess)):
            if str_guess[idx] == str_secret[idx]:
                bulls += 1
            else:
                rem_idxs.add(idx)
                rem_secret += str_secret[idx]

        # calculate cows in the remaining portion
        for idx in rem_idxs:
            if str_guess[idx] in rem_secret:
                assert str_guess[idx] != str_secret[idx]
                cows += 1
        
        assert (bulls + cows) <= 3
        return (bulls, cows)
