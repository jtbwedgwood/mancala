import math
import random
import time

class Mancala():
    def __init__(self, holes=6, init_beads=4):
        self.holes = holes
        self.init_beads = init_beads
        self.player = 0
        self.winner = None
        self.board = [[self.init_beads] * self.holes, [self.init_beads] * self.holes]
        self.piles = [0, 0]

    def __repr__(self):
        return display_board(self.board, self.piles)

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    @classmethod
    def available_actions(cls, state):
        board, player = state

        # Action (i, j) represents starting with the jth pile on player i's side
        if all(beads == 0 for beads in board[player]):
            open_side = Mancala.other_player(player)
        else:
            open_side = player
        return {(open_side, hole) for hole, beads in enumerate(board[open_side]) if beads != 0}

    def switch_player(self):
        self.player = Mancala.other_player(self.player)

    def beads(self, position):
        return self.board[position[0]][position[1]]

    def empty(self, position):
        self.board[position[0]][position[1]] = 0

    def incr(self, position):
        if position == (self.player, self.holes):
            self.piles[self.player] += 1
        else:
            self.board[position[0]][position[1]] += 1

    def move(self, action):
        remaining_beads = self.beads(action)

        # Empty current space
        self.empty(action)

        # Go around
        current_pos = action
        while remaining_beads:
            side, hole = current_pos
            if side == self.player and hole == self.holes:
                current_pos = (Mancala.other_player(self.player), 0)
            elif side == Mancala.other_player(self.player) and hole == self.holes - 1:
                current_pos = (self.player, 0)
            else:
                current_pos = (side, hole + 1)
            self.incr(current_pos)
            remaining_beads -= 1

        # Check for winner
        if sum(self.piles) == self.holes * self.init_beads * 2:
            self.winner = max([0, 1], key=lambda x: self.piles[x])

        # Continue unless a stopping point is reached
        if current_pos != (self.player, self.holes) and self.beads(current_pos) != 1:
            self.move(current_pos)

# Copied from my implementation of MancalaAI for the CS50 AI in Python course
class MancalaAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple containing the board state and the current player
         - `action` is a tuple `(i, j)` for an action
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """

        # Return 0 if the key (state, action) is invalid
        try:
            return self.q[(deep_tuple(state[0]), state[1]), action]
        except KeyError:
            return 0

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estimate of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """

        # New value = reward + future_rewards
        self.q[(deep_tuple(state[0]), state[1]), action] = old_q + self.alpha * (reward + future_rewards - old_q)

    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """

        # Get list of rewards using available_actions classmethod
        rewards = [self.get_q_value(state, action) for action in Mancala.available_actions(state)]
        return max(rewards) if rewards else 0

    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """

        # Using random.random() to get a real number between 0 and 1
        if epsilon and random.random() < self.epsilon:

            # Using the available_actions classmethod
            return random.choice(list(Mancala.available_actions(state)))

        # Otherwise get the best one
        else:
            return max(Mancala.available_actions(state), key=lambda x: self.get_q_value(state, x))


def display_board(board, piles):

    # Player 0 goes up the right side, player 1 goes down the left side
    stars = [
        ("*" * i).rjust(max(board[1])) + " " + ("*" * j)
        for i, j in zip(board[1], board[0][::-1])
    ]
    scores = [
        f"Player 0: {piles[0]}",
        f"Player 1: {piles[1]}"
    ]
    return "\n".join(stars + scores)

def deep_tuple(lst):
    return tuple([tuple(item) for item in lst])

def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    player = MancalaAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Mancala()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            state = (game.board.copy(), game.player)
            action = player.choose_action((game.board, game.player))

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = (game.board.copy(), game.player)

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )
            game.switch_player()

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Mancala()

    # Game loop
    while True:

        # Print contents of piles
        print(game)

        # Compute available actions
        available_actions = Mancala.available_actions((game.board, game.player))
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                side = int(input("Choose Side: "))
                hole = int(input("Choose Hole: "))
                if (side, hole) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            side, hole = ai.choose_action((game.board, game.player), epsilon=False)
            print(f"AI chose to start with hole {hole} on side {side}.")

        # Make move
        game.move((side, hole))
        game.switch_player()

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
