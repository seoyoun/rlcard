from rlcard.games.gin_rummy.player import GinRummyPlayer

player = GinRummyPlayer(player_id=0, np_random=None)
print(f'Player Initialized: {player}')
player.update_meld_count()
print(f'Meld Count: {player.previous_meld_count}')
