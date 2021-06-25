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
from const import MAX_REPLAY_SIZE, POS_NUMBERS


class Tournament(object):

    def __init__(self) -> None:
        self.replay_buffer = deque(maxlen=MAX_REPLAY_SIZE)
    
    def play(self, style='exploratory'):
        if style == 'exploratory':
            pass
        elif style == 'competition':
            pass
    
    def train(self):
        pass


class Game(object):

    FULL_SCORE = (3, 0)
    def __init__(self) -> None:
        pass

    def start_game(self, players):
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
        while result is None:
            # play a round and compare scores
            scores = self._play_round()
            if scores[0] == Game.FULL_SCORE:
                result += 1
            if scores[1] == Game.FULL_SCORE:
                result -= 1

            turns += 1
            players[0].set_score(scores[0])
            players[1].set_score(scores[1])

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
