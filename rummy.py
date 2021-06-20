"""
Only pick cards from closed deck
A = 1, Ace can only be with 2, 3, ...
"""

from deck_of_cards.deck_of_cards import Card, DeckOfCards

from copy import deepcopy


def create_empty_deck():
    deck_obj = DeckOfCards()
    deck_obj.deck.clear()

    return deck_obj


class Game(object):

    CARDS_TO_DEAL = 13
    JOKERS = [Card((4, 14))] * 4

    def __init__(self, players) -> None:
        self.players = players
        self.NUM_PLAYERS = len(players)
        assert self.NUM_PLAYERS == 4

        self.player_decks = [
            create_empty_deck() for _ in range(self.NUM_PLAYERS)
        ]

    def start_match(self):
        # create closed deck with 2 decks of cards and add 4 jokers
        # create open deck that is empty
        self.closed_deck = DeckOfCards()
        self.closed_deck.add_deck()
        self.open_deck = create_empty_deck()

        assert len(self.closed_deck.deck) == 104
        assert len(Game.JOKERS) == 4
        self.closed_deck.deck.extend(Game.JOKERS)  # hacky fix

        # deal cards to players
        self._deal_init_cards()
        
        # pick joker and stack card
        stack_card = self.closed_deck.give_first_card()
        self.open_deck.take_card(stack_card)
        self.joker_rank = self.closed_deck.give_random_card().rank

        # if joker gets picked, Ace is joker rank
        if self.joker_rank == 14:
            self.joker_rank = 1
        
        # play rounds till winner emerges
        game_over = False
        while not game_over:
            game_over, winner = self._play_round()
        
        print(f"Player {winner} has won the game!")
    
    def _deal_init_cards(self):
        # shuffle the deck
        self.closed_deck.shuffle_deck()

        for idx, card in enumerate(self.closed_deck.deck):
            if idx < self.NUM_PLAYERS * Game.CARDS_TO_DEAL:
                # deal first card to next player
                card_to_be_dealt = self.closed_deck.give_first_card()

                self.player_decks[idx % self.NUM_PLAYERS].deck.append(
                    card_to_be_dealt
                )
            else:
                break
    
    def _play_round(self):
        # NOTE the game interface exposes certain parameters to the player
        # assuming that the players are not malicious and play the game
        # "fairly".
        # It is interesting to note that Python does allow access to
        # class attributes. Therefore, a malicious player could
        # cheat in almost any design of code.
        # (to the best of the author's knowledge)

        game_over = False
        for idx, player in enumerate(self.players):
            # give the next card to the player and update stack card
            # to the discarded card from player
            card_to_be_picked = self.closed_deck.give_first_card()
            stack_card = player.play(
                card_to_be_picked,
                self.player_decks[idx].deck,
                self.joker_rank
            )
            self.open_deck.take_card(stack_card)

            # check if the player won
            score = self._get_score(idx)
            game_over = (score == 0)
            return (game_over, idx if game_over else None)

    def _get_score(self, player_idx):
        num_sequences, num_triplets = 0, 0

        dup_deck = deepcopy(self.player_decks[player_idx]).order_deck()
        orig_set_limits = self._get_orig_set(dup_deck)

    def _get_orig_set(self, deck):

        # iterate reverse to find biggest sequences first
        for idx in range(len(deck) - 1, -1, -1):
            card = deck[idx]
            # skip joker card
            if card.suit == 4:
                continue

            seq_suit, curr_rank = card.suit, card.rank
            seq_idx = 1
            temp_card = deck[idx - seq_idx]
            # print("For", idx, seq_suit, curr_rank)

            # for each card, try to match a sequence
            while temp_card.rank >= curr_rank - 1:
                # print("While", seq_suit, curr_rank, seq_idx)
                # print("While", temp_card.suit, temp_card.rank)
                if temp_card.suit == seq_suit \
                  and temp_card.rank == curr_rank - 1:
                    curr_rank -= 1
                    # print("Dec", curr_rank)
                else:
                    if seq_idx >= 3:
                        return (idx - seq_idx, idx)
                    else:
                        break

                seq_idx += 1
                temp_idx = idx - seq_idx
                if temp_idx < 0:
                    return (temp_idx + 1, idx)
                else:
                    temp_card = deck[temp_idx]
        else:
            return None