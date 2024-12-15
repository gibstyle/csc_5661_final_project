import numpy as np
from abc import ABC, abstractmethod

class Player(ABC):
    """
    A class to hold the player for the Euchre game.
    """
    def __init__(self, id: int, team_id: int, train: bool) -> None:
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

        self.train = train
        self.data_t = {}

    @abstractmethod
    def choose_action(self, state):
        """
        Abstract method to be implemented by subclasses.
        Given a state, this function returns an action.
        """
        pass

    @abstractmethod
    def train_update(self, state_t_1, reward_t_1, count):
        """
        Abstract method to be implemented by subclasses.
        Given a state, reward, and count, the agent is updated
        """
        pass

    @abstractmethod
    def choose_trump(self, state, trump, suits, count):
        """
        Abstract method to be implemented by subclasses.
        Given a state, this function returns an action.
        """
        pass

    @abstractmethod
    def discard_card(self, state):
        """
        Abstract method to be implemented by subclasses.
        Given a state, this function takes a 6 card hand and discards a card.
        """

    def get_trick_hand(self, state: dict):
        lead_suit = state.get('lead_suit', None)
        if lead_suit is None:
            raise ValueError("Error with lead suit.")

        hand = state.get('hand', [])
        if not hand:
            raise ValueError("No cards in hand.")
        
        if lead_suit and lead_suit != -1:
            new_hand = [card for card in hand if lead_suit in card]
            hand = new_hand if len(new_hand) != 0 else hand

        return hand

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
        self.current_card = card
        try:
            self.hand.remove(card)
        except Exception as e:
            print(e)

    # def choose_card(self, random=False):
    #     """
    #     Determine the card to choose for the current trick.

    #     Parameters
    #     ----------
    #     random : bool
        
    #     """
    #     if len(self.hand) != 1:  # if there are more than 1 card in the hand
    #         if random:  # choose a random card from the hand
    #             card = np.random.choice(self.hand)
    #         self.update_hand(card)
    #     else:  # only 1 card left, have to play what is left
    #         card = self.hand.pop(0)
    #     return card
    
    # def choose_card_agent(self, agent):
    #     """
    #     Determine the card to choose for the current trick.

    #     Parameters
    #     ----------
    #     agent : agent to make decision
        
    #     """
    #     #TODO
    #     pass
