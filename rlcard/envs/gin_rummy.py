'''
    File name: envs/gin_rummy.py
    Author: William Hale
    Date created: 2/12/2020
'''
import numpy as np
from collections import OrderedDict

from rlcard.envs import Env

class GinRummyEnv(Env):
    ''' GinRummy Environment
    '''
    def __init__(self, config):
        from rlcard.games.gin_rummy.utils.move import ScoreSouthMove
        from rlcard.games.gin_rummy.utils import utils
        from rlcard.games.gin_rummy import Game
        self._ScoreSouthMove = ScoreSouthMove
        self._utils = utils

        self.name = 'gin-rummy'
        self.game = Game()

        # self.game.env = self #added

        super().__init__(config=config)
        self.state_shape = [[5, 52] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):  # 200213 don't use state ???
        ''' Encode state

        Args:
            state (dict): dict of original state

        Returns:
            numpy array: 5 * 52 array
                         5 : current hand (1 if card in hand else 0)
                             top_discard (1 if card is top discard else 0)
                             dead_cards (1 for discards except for top_discard else 0)
                             opponent known cards (likewise)
                             unknown cards (likewise)  # is this needed ??? 200213
        '''
        if self.game.is_over():
            obs = np.array([self._utils.encode_cards([]) for _ in range(5)])
            extracted_state = {'obs': obs, 'legal_actions': self._get_legal_actions()}
            extracted_state['raw_legal_actions'] = list(self._get_legal_actions().keys())
            extracted_state['raw_obs'] = obs
        else:
            discard_pile = self.game.round.dealer.discard_pile
            stock_pile = self.game.round.dealer.stock_pile
            top_discard = [] if not discard_pile else [discard_pile[-1]]
            dead_cards = discard_pile[:-1]
            current_player = self.game.get_current_player()
            opponent = self.game.round.players[(current_player.player_id + 1) % 2]
            known_cards = opponent.known_cards
            unknown_cards = stock_pile + [card for card in opponent.hand if card not in known_cards]
            hand_rep = self._utils.encode_cards(current_player.hand)
            top_discard_rep = self._utils.encode_cards(top_discard)
            dead_cards_rep = self._utils.encode_cards(dead_cards)
            known_cards_rep = self._utils.encode_cards(known_cards)
            unknown_cards_rep = self._utils.encode_cards(unknown_cards)
            rep = [hand_rep, top_discard_rep, dead_cards_rep, known_cards_rep, unknown_cards_rep]
            obs = np.array(rep)
            extracted_state = {'obs': obs, 'legal_actions': self._get_legal_actions(), 'raw_legal_actions': list(self._get_legal_actions().keys())}
            extracted_state['raw_obs'] = obs
        return extracted_state

    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        # determine whether game completed all moves
        is_game_complete = False
        if self.game.round:
            move_sheet = self.game.round.move_sheet
            if move_sheet and isinstance(move_sheet[-1], self._ScoreSouthMove):
                is_game_complete = True
        payoffs = [0, 0] if not is_game_complete else self.game.judge.scorer.get_payoffs(game=self.game)
        return np.array(payoffs)

    def _decode_action(self, action_id):  # FIXME 200213 should return str
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action (ActionEvent): the action that will be passed to the game engine.
        '''
        return self.game.decode_action(action_id=action_id)

    def _get_legal_actions(self):
        ''' Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_actions = self.game.judge.get_legal_actions()
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)
    
    def get_winner(self):

        player_0_deadwood = self.game.round.players[0].get_deadwood_count()
        player_1_deadwood = self.game.round.players[1].get_deadwood_count()


        if player_0_deadwood < player_1_deadwood:

            return 0
        else:
            return 1


    # #added
    # def _get_rewards(self):
    #     ''' Compute dense rewards for the current game state.

    #     Returns:
    #         rewards (list): A list of rewards for each player.
    #     '''
    #     rewards = [0, 0]  # Initialize rewards for both players

    #     if self.game.round:  # Ensure the round exists
    #         for player_id in [0, 1]:
    #             player = self.game.round.players[player_id]

    #             # Example 1: Reward deadwood reduction
    #             current_deadwood = player.get_deadwood_count()
    #             previous_deadwood = getattr(player, 'previous_deadwood_count', None)
    #             if previous_deadwood is not None:
    #                 rewards[player_id] += previous_deadwood - current_deadwood
                
    #             # Example 2: Penalize discarding valuable cards for the opponent
    #             last_discard = getattr(player, 'last_discard', None)
    #             opponent = self.round.players[1 - player_id]
    #             if last_discard and last_discard in opponent.hand:
    #                 rewards[player_id] -= 1  # Penalize for discarding a useful card

    #             # Example 3: Reward forming melds
    #             current_melds = len(player.get_meld_count())
    #             previous_melds = getattr(player, 'previous_meld_count', None)
    #             if previous_melds is not None:
    #                 rewards[player_id] += (current_melds - previous_melds) * 2  # Higher reward for melds
                
    #             # Update stored metrics for the next step
    #             player.previous_deadwood_count = current_deadwood
    #             player.previous_meld_count = current_melds
    #             player.last_discard = self.round.last_discard_pile[-1] if self.round.last_discard_pile else None

    #     return np.array(rewards)
