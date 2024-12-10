from .player import Player

import random


class RandomAgent(Player):
    """
    
    """
    def __init__(self, id: int, team_id: int, train: bool, name="Random Agent"):
        super().__init__(id=id, team_id=team_id, train=train)
        self.name = name

    def choose_action(self, state: dict):
        hand = self.get_trick_hand(state=state)

        action = random.choice(hand)
        self.update_hand(action)
        return action
    
    def train_update(self, state_t_1, reward_t_1, count):
        pass