from .player import Player

import random
import copy


class RandomAgent(Player):
    """
    
    """
    def __init__(self, id: int, team_id: int, train: bool, config: dict, name="Random Agent"):
        super().__init__(id=id, team_id=team_id, train=train)
        self.name = name
        self.config = config

    def choose_action(self, state: dict):
        if state['phase'] == 0:  # bidding
            action = random.choice([-3, -2])  # -3 for pass, -2 for "order up"
        
        elif state['phase'] == 1:  # remove card
            start_actions = self.hand + [state['top_card']]
            actions = [self.config['card_values'][card] for card in start_actions]
            action = random.choice(actions)
            action_update = start_actions[actions.index(action)]
            if action_update != state['top_card']:
                self.update_hand(action_update)  # remove the action (card) from the hand
                self.hand.append(state['top_card'])  # add in the top card to the hand

        elif state['phase'] == 2: # trump selection
            actions = state['suits'] if state['dealer'] else [-1] + state['suits']
            action = random.choice(actions)

        else:
            hand = self.get_trick_hand(state)
            actions = [self.config['card_values'][card] for card in hand]
            action = random.choice(actions)
            self.update_hand(hand[actions.index(action)])

        return action
    
    def train_update(self, state_t_1, reward_t_1, count):
        pass

    def choose_trump(self, state, trump, suits, count):
        current_suits = copy.deepcopy(suits)
        current_suits.remove(trump)
        if count <= 4:
            if random.random() < 0.7:
                return self.calls[1]
            else:
                return self.calls[0]
        elif count <= 7:
            if random.random() < 0.7:
                return self.calls[1]
            else:
                return random.choice(current_suits)
        elif count == 8:
            return random.choice(current_suits)

    
    def discard_card(self):
        self.hand.remove(random.choice(self.hand))
