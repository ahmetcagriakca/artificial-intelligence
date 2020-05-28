from sample_players import DataPlayer
import random


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least
        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.
        See RandomPlayer and GreedyPlayer in sample_players for more examples.
        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        
        # from isolation import DebugState
        # print(DebugState.from_state(state))
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            # iterative deepening implemented 
            depth = 1
            while 1:
                self.queue.put(self.alpha_beta_search(state, depth=depth))
                depth += 1
                print(depth)

    def alpha_beta_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in state.actions():
            value = self.min_value(state.result(action), alpha, beta, depth)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = action
        return best_move

    def min_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state)
        value = float("inf")
        for action in state.actions():
            value =  min(value,self.max_value(state.result(action), alpha, beta, depth - 1))
            if value <= alpha:
                return value
            else:
                beta = min(beta, value)
        return value

    def max_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state)
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.min_value(state.result(action), alpha, beta, depth - 1))
            if value >= beta:
                return value
            else:
                alpha = max(alpha, value)
        return value

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

# inheriting from CustomPlayer
class BaselinePlayer(CustomPlayer):
    pass


class DefensivePlayer(CustomPlayer):
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        return len(own_liberties) 

        
class AggressivePlayer(CustomPlayer):
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - 2*len(opp_liberties)

# inheriting from CustomPlayer
# Use heuristic with location differences
class LocationStrategistPlayer(CustomPlayer):
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        heuristic = 0
        if own_loc-opp_loc>= 0 :
            heuristic = own_loc-opp_loc
        else:
            heuristic= -1 * (own_loc-opp_loc)
        return heuristic