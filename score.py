import re

from player import Player


class Score:
    """
    A class to use as a helper to score winners of tricks and hands.
    """
    def __init__(self):
        """
        The constructor for the class.
        """
        self.rank_order = {"J": 3, "A": 6, "K": 5, "Q": 4, "10": 2, "9": 1}  # the rank order of the cards
        
        # the left bower (off-suit jack)
        self.left_bower = {
            "hearts": "diamonds",
            "diamonds": "hearts",
            "spades": "clubs",
            "clubs": "spades"
        }
        
        self.suits = r'[♠♥♦♣]'

    def score_trick(self, players: list[Player], trump_suit: str, lead_suit: str) -> int:
        """
        Calculate the winner of the trick.

        Example:
        If trump suit was Hearts, the ranking of trump cards would be:
            1. J♦ (Right Bower)
            2. J♥ (Left Bower)
            3. A♦
            4. K♦
            5. Q♦
            6. 10♦ 
            7. 9♦

        Parameters
        ----------
        players : list[Player]
            The list of players in the game.
        trump_suit : str
            The trump suit for the trick.
        lead_suit : str
            The lead suit for the trick.

        Returns
        -------
        highest_player_id : int
            The player id who won the trick.
        """
        left_bower_suit = self.left_bower.get(trump_suit)

        highest_card = -1
        highest_player_id = None

        # loop through each player to determine highest card in trick
        for player in players:
            rank = re.split(self.suits, player.current_card)[0] # get rank
            suit = re.findall(self.suits, player.current_card)[0]  # get suit

            rank_value = self.rank_order[rank]
            if suit == trump_suit or (rank == 'J' and suit == left_bower_suit):  # is trump suite
                # add highest rank card to rank to ensure it is higher than other cards
                card_rank = rank_value + 6
                # see if right or left bower
                if rank == 'J':
                    card_rank = card_rank + 5 if suit == trump_suit else card_rank + 4
            elif suit == lead_suit:  # non trump card that is lead suit
                card_rank = rank_value  
            else:  # non trump card and not lead suit
                card_rank = 0

            # compare to see if new highest card found
            if highest_card == -1 or card_rank > highest_card:
                highest_card = card_rank
                highest_player_id = player.id

        return highest_player_id

    def score_hand(self, players: list[Player], solo_call=False) -> dict:
        """
        Calculate the points for the makers or the defenders.

        Parameters
        ----------
        players : list[Player]
            The list of players in the game.
        solo_call : bool (optional)
            If the player on the maker chose to go solo, default = False.

        Returns
        -------
        dict
            A dictionary containing the scores of the makers and the defenders.
        """
        # get makers and defenders points from each partnership
        makers_tricks_won = 0
        defenders_tricks_won = 0
        for player in players:
            if player.trick_team == 'makers':
                makers_tricks_won += player.points
            elif player.trick_team == 'defenders':
                defenders_tricks_won += player.points

        # calculate each teams points
        makers_points = 0
        defenders_points = 0
        if makers_tricks_won == 5:
            makers_points = 4 if solo_call else 2
        elif makers_tricks_won >= 3:
            makers_points = 2 if solo_call else 1
        else:
            defenders_points = 2
            
        return {
            'makers': makers_points,
            'defenders': defenders_points
        }  
    