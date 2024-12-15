import re

from players.player import Player


class Score:
    """
    A class to use as a helper to score winners of tricks and hands.
    """
    def __init__(self):
        """
        The constructor for the class.
        """
        self.rank_order = {
            "A": 6,
            "K": 5,
            "Q": 4,
            "J": 3,
            "10": 2,
            "9": 1
        }  # the rank order of the cards
        
        # the left bower (off-suit jack)
        self.left_bower = {
            "♥": "♦",
            "♦": "♥",
            "♠": "♣",
            "♣": "♠"
        }
        
        self.suits = r'[♠♥♦♣]'

    def calculate_hand_strength(self, hand, trump_suit):
        """
        Calculate the strength of a given hand based on the trump suit.
        
        Parameters:
            hand (list): A list of cards, e.g., ["J♠", "9♠", "A♦"].
            trump_suit (str): The trump suit, e.g., "♠".
            
        Returns:
            int: The calculated hand strength.
        """
        strength = 0
        
        for card in hand:
            rank = card[:-1]  # Extract rank (e.g., "J" from "J♠")
            suit = card[-1]   # Extract suit (e.g., "♠" from "J♠")
            
            # Check if the card is a trump card
            if suit == trump_suit:
                strength += self.rank_order.get(rank, 0) + 5  # Boost for trump cards
            
            # Check if the card is the left bower (off-suit jack of the same color)
            elif rank == "J" and suit == self.left_bower.get(trump_suit, ""):
                strength += self.rank_order.get(rank, 0) + 4  # Slightly less than a trump Jack
            
            # Non-trump cards
            else:
                strength += self.rank_order.get(rank, 0)
        
        return strength

    def score_trick(self, actions: dict, trump_suit: str, lead_suit: str) -> int:
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
        actions : dict
            The actions the players took where key = player id, value = card (action)
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
        for id, action in actions.items():
            rank = re.split(self.suits, action['action'])[0] # get rank
            suit = re.findall(self.suits, action['action'])[0]  # get suit

            card_rank = self.rank_order[rank]
            if suit == trump_suit or (rank == 'J' and suit == left_bower_suit):  # is trump suit
                # add highest rank card to rank to ensure it is higher than other cards
                card_rank += 7
                # see if right or left bower
                if rank == 'J':
                    card_rank = card_rank + 5 if suit == trump_suit else card_rank + 4
            elif suit == lead_suit:  # non trump card that is lead suit
                card_rank = self.rank_order[rank]  
            else:  # non trump card and not lead suit
                card_rank = 0

            # compare to see if new highest card found
            if highest_card == -1 or card_rank > highest_card:
                highest_card = card_rank
                highest_player_id = id

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
    