import torch

from mastermind import Trainer

from const import (
    NUM_TRAIN_LOOPS, MAX_LOSS_FRAC, NUM_COMP_MATCHES,
    EPSILON, E_DECAY, E_MIN
)

from copy import deepcopy

if __name__ == "__main__":
    T = Trainer()
    eps = EPSILON

    while True:
        # simulate matches between best nets
        T.simulate('exploratory')

        # train loops
        best_copy = deepcopy(T.best_nn)
        for t_idx in range(NUM_TRAIN_LOOPS):
            T.train()

        # despite a bit of an unconventional naming, latest_nn is the
        # newly trained nn. best_nn is still the one before training
        T.latest_nn = T.best_nn
        T.best_nn = best_copy

        # compete latest and best
        with torch.no_grad():
            results = T.simulate('competition')

        # NOTE assuming best_nn is player_one
        # and latest_nn is player_two
        losses = results.unique(return_counts=True)[1][-1]
        if (losses.item() / NUM_COMP_MATCHES) < MAX_LOSS_FRAC:
            T.best_nn = T.latest_nn

        eps = max(E_MIN, eps * E_DECAY)
