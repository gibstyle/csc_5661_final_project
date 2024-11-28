import numpy as np

class Player:
    """
    A class to hold the player for the Euchre game.
    """
    def __init__(self, id: int, team_id: int) -> None:
        """
        The constructor for the class.

        Parameters
        ----------
        id : int
            The id of the player (1 - 4).
        team_id : int
            The team id for which team the player is on to idenfity their partner.
        """
        self.id = id
        self.team_id = team_id
        
        self.hand: list = []  # the current hand of the player for the hand
        self.is_dealer: bool = False  # True if they are the dealer of current trick, False if not
        self.points: int = 0  # the number of tricks won for the hand
        self.current_card: str = ''  # the card the player used for the current trick
        self.trick_team: str = ''  # either makers or defenders for the trick

    def set_hand(self, cards: list) -> None:
        """
        Set the hand for the player after dealing the cards.

        Parameters
        ----------
        hand : list
            The set of cards to set as the current hand.
        """
        self.hand = cards

    def update_hand(self, card: str):
        """
        Remove the card from the hand.

        Parameters
        ----------
        card : str
            The card to remove from the hand.
        """
        try:
            self.hand.remove(card)
        except Exception as e:
            print(e)

    def choose_card(self, random=False):
        """
        Determine the card to choose for the current trick.

        Parameters
        ----------
        random : bool
        
        """
        if len(self.hand) != 1:  # if there are more than 1 card in the hand
            if random:  # choose a random card from the hand
                card = np.random.choice(self.hand)
            else:  # choose the best card from the hand
                card = 0  # TODO: add function to get best card to choose
            self.update_hand(card)
        else:  # only 1 card left, have to play what is left
            card = self.hand.pop(0)
        return card
