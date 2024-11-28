class Player:
    def __init__(self, id):
        self.id = id
        self.hand = []

        self.is_dealer: bool = False
        self.points: int = 0
        self.current_card: str = ''
        self.team: str = ''  # either makers or defenders

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
