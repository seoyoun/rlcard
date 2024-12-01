'''
    File name: gin_rummy/game.py
    Author: William Hale
    Date created: 2/12/2020
'''

import numpy as np

from .player import GinRummyPlayer
from .round import GinRummyRound
from .judge import GinRummyJudge
from .utils.settings import Settings, DealerForRound

from .utils.action_event import *


class GinRummyGame:
    ''' Game class. This class will interact with outer environment.
    '''

    def __init__(self, allow_step_back=False):
        '''Initialize the class GinRummyGame
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.judge = GinRummyJudge(game=self)
        self.settings = Settings()
        self.actions = None  # type: List[ActionEvent] or None # must reset in init_game
        self.round = None  # round: GinRummyRound or None, must reset in init_game
        self.num_players = 2
        self.ender = None

    def init_game(self):
        ''' Initialize all characters in the game and start round 1
        '''
        dealer_id = self.np_random.choice([0, 1])
        if self.settings.dealer_for_round == DealerForRound.North:
            dealer_id = 0
        elif self.settings.dealer_for_round == DealerForRound.South:
            dealer_id = 1
        self.actions = []
        self.round = GinRummyRound(dealer_id=dealer_id, np_random=self.np_random)
        for i in range(2):
            num = 11 if i == 0 else 10
            player = self.round.players[(dealer_id + 1 + i) % 2]
            self.round.dealer.deal_cards(player=player, num=num)
        current_player_id = self.round.current_player_id
        state = self.get_state(player_id=current_player_id)
        return state, current_player_id

    def step(self, action: ActionEvent):
        ''' Perform game action and return next player number, and the state for next player
        '''
        bonus_reward = 0
        current_player = self.round.current_player_id
        if isinstance(action, ScoreNorthPlayerAction):
            self.round.score_player_0(action)
        elif isinstance(action, ScoreSouthPlayerAction):
            self.round.score_player_1(action)


        elif isinstance(action, DrawCardAction):
            self.round.draw_card(action)
            # print("draw card")
        elif isinstance(action, PickUpDiscardAction):
            self.round.pick_up_discard(action)
            # print("draw card - known")


        elif isinstance(action, DiscardAction):
            self.round.discard(action)
            # print("discard")
            bonus_reward += self._get_reward_dw(current_player)

        elif isinstance(action, DeclareDeadHandAction):
            self.ender = 1
            self.round.declare_dead_hand(action)
            bonus_reward -= 1
        elif isinstance(action, GinAction):
            self.ender = current_player
            bonus_reward += 1
            # print("GIN",current_player)
            self.round.gin(action, going_out_deadwood_count=self.settings.going_out_deadwood_count)
        elif isinstance(action, KnockAction):
            self.ender = current_player
            bonus_reward += 1
            # print("Knock",current_player)
            self.round.knock(action)
        else:
            raise Exception('Unknown step action={}'.format(action))
        self.actions.append(action)
        next_player_id = self.round.current_player_id
        next_state = self.get_state(player_id=next_player_id)

        # added 
        # reward = self._get_reward(current_player)

        reward = self._get_reward_melds(current_player)
        # print("meld reward: ", reward, "bonus:", bonus_reward)
        reward += bonus_reward
        # print("total reward: ", reward)
        return next_state, next_player_id, reward


    def step_2(self, action: ActionEvent):
        ''' Perform game action and return next player number, and the state for next player
        '''
        bonus_reward = 0
        current_player = self.round.current_player_id
        if isinstance(action, ScoreNorthPlayerAction):
            self.round.score_player_0(action)
        elif isinstance(action, ScoreSouthPlayerAction):
            self.round.score_player_1(action)


        elif isinstance(action, DrawCardAction):
            self.round.draw_card(action)
            # print("draw card")
        elif isinstance(action, PickUpDiscardAction):
            self.round.pick_up_discard(action)
            # print("draw card - known")


        elif isinstance(action, DiscardAction):
            self.round.discard(action)
            # print("discard")
            bonus_reward += self._get_reward_dw(current_player)

        elif isinstance(action, DeclareDeadHandAction):
            self.round.declare_dead_hand(action)
            bonus_reward -= 2
        elif isinstance(action, GinAction):
            bonus_reward += 2
            # print("GIN")
            self.round.gin(action, going_out_deadwood_count=self.settings.going_out_deadwood_count)
        elif isinstance(action, KnockAction):
            bonus_reward += 2
            # print("Knock")
            self.round.knock(action)
        else:
            raise Exception('Unknown step action={}'.format(action))
        self.actions.append(action)
        next_player_id = self.round.current_player_id
        next_state = self.get_state(player_id=next_player_id)

        # added 
        # reward = self._get_reward(current_player)

        reward = self._get_reward_melds(current_player)
        # print("meld reward: ", reward, "bonus:", bonus_reward)
        reward += bonus_reward
        # print("total reward: ", reward)
        return next_state, next_player_id, reward



    def _get_reward(self, player_id):
        """
        Calculate dense rewards based on the change in deadwood count and meld formations.
        
        Args:
            player_id (int): The player ID.
            previous_deadwood_count (int): The deadwood count before the last action.
            current_deadwood_count (int): The deadwood count after the last action.
            melds_formed (bool): Whether the player formed a meld during the action.

        Returns:
            float: The reward for the player.
        """
        player = self.round.players[player_id]

        reward = 0.0

        current_deadwood_count = player.get_deadwood_count()
        previous_deadwood_count = player.get_previous_deadwood_count()
        current_melds = player.get_meld_count()
        previous_melds = player.get_previous_meld_count()

        # # Reward for decreasing deadwood
        # if current_deadwood_count < previous_deadwood_count:
        #     reward += 0.1  # Small reward for reducing deadwood

        # # Penalize for increasing deadwood
        # elif current_deadwood_count > previous_deadwood_count:
        #     reward -= 0.1  # Small penalty for increasing deadwood

        # Reward for forming a meld (either set or run)
        if current_melds > previous_melds:
            reward += 0.3  # Reward for forming a meld (you can adjust this value)



        return reward

    # melds reward, draw & discard
    def _get_reward_melds(self, player_id):
        """
        reward for forming a meld: 
        punish for breaking a meld:
        """
        player = self.round.players[player_id]

        
        current_melds = player.get_meld_count()
        previous_melds = player.get_previous_meld_count()
        if current_melds > previous_melds:
            reward = 0.3
        elif current_melds == previous_melds:
            reward = 0
        else:
            reward = -0.3
        return reward
    
    # deadwood reward, use in discard
    def _get_reward_dw(self, player_id):
        player = self.round.players[player_id]

        reward = 0.0

        current_deadwood_count = player.get_deadwood_count()
        previous_deadwood_count = player.get_previous_deadwood_count()
        # print(previous_deadwood_count,current_deadwood_count)
        reward = (previous_deadwood_count - current_deadwood_count)/100
        return reward
    

    # #added
    # def _get_rewards(self):
    #     ''' Compute dense rewards for the current game state.

    #     Returns:
    #         rewards (list): A list of rewards for each player.
    #     '''
    #     rewards = [0, 0]  # Initialize rewards for both players

    #     if self.round:  # Ensure the round exists
    #         for player_id in [0, 1]:
    #             player = self.round.players[player_id]

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
    #             current_melds = player.get_meld_count()
    #             previous_melds = getattr(player, 'previous_meld_count', None)
    #             if previous_melds is not None:
    #                 rewards[player_id] += (current_melds - previous_melds) * 2  # Higher reward for melds
                
    #             # Update stored metrics for the next step
    #             player.previous_deadwood_count = current_deadwood
    #             player.previous_meld_count = current_melds
    #             player.last_discard = self.round.last_discard_pile[-1] if self.round.last_discard_pile else None

    #     return np.array(rewards)





    def step_back(self):
        ''' Takes one step backward and restore to the last state
        '''
        raise NotImplementedError

    def get_num_players(self):
        ''' Return the number of players in the game
        '''
        return 2

    def get_num_actions(self):
        ''' Return the number of possible actions in the game
        '''
        return ActionEvent.get_num_actions()

    def get_player_id(self):
        ''' Return the current player that will take actions soon
        '''
        return self.round.current_player_id

    def is_over(self):
        ''' Return whether the current game is over
        '''
        return self.round.is_over

    def get_current_player(self) -> GinRummyPlayer or None:
        return self.round.get_current_player()

    def get_last_action(self) -> ActionEvent or None:
        return self.actions[-1] if self.actions and len(self.actions) > 0 else None

    def get_state(self, player_id: int):
        ''' Get player's state

        Return:
            state (dict): The information of the state
        '''
        state = {}
        if not self.is_over():
            discard_pile = self.round.dealer.discard_pile
            top_discard = [] if not discard_pile else [discard_pile[-1]]
            dead_cards = discard_pile[:-1]
            last_action = self.get_last_action()
            opponent_id = (player_id + 1) % 2
            opponent = self.round.players[opponent_id]
            known_cards = opponent.known_cards
            if isinstance(last_action, ScoreNorthPlayerAction) or isinstance(last_action, ScoreSouthPlayerAction):
                known_cards = opponent.hand
            unknown_cards = self.round.dealer.stock_pile + [card for card in opponent.hand if card not in known_cards]
            state['player_id'] = self.round.current_player_id
            state['hand'] = [x.get_index() for x in self.round.players[self.round.current_player_id].hand]
            state['top_discard'] = [x.get_index() for x in top_discard]
            state['dead_cards'] = [x.get_index() for x in dead_cards]
            state['opponent_known_cards'] = [x.get_index() for x in known_cards]
            state['unknown_cards'] = [x.get_index() for x in unknown_cards]
        return state

    @staticmethod
    def decode_action(action_id) -> ActionEvent:  # FIXME 200213 should return str
        ''' Action id -> the action_event in the game.

        Args:
            action_id (int): the id of the action

        Returns:
            action (ActionEvent): the action that will be passed to the game engine.
        '''
        return ActionEvent.decode_action(action_id=action_id)
