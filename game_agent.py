import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def get_moves(game, loc):
    """Generate the list of possible moves for an L-shaped motion (like a
    knight in chess).
    """
    r, c = loc
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    valid_moves = [(r + dr, c + dc) for dr, dc in directions
                   if game.move_is_legal((r + dr, c + dc))]
    random.shuffle(valid_moves)
    return valid_moves


def custom_score(game, player):
    """Returned score is the ratio between the number of available legal moves
    for the computer player plus a look-ahead count to the computer players
    available moves in future to the weighted number of available legal moves for
    the opponenent.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opp_legal_moves = game.get_legal_moves(game.get_opponent(player))
    if not opp_legal_moves:
        return float("inf")

    opp_moves = len(opp_legal_moves)
    for m in opp_legal_moves:
        opp_moves += len(get_moves(game, m))

    own_legal_moves = game.get_legal_moves(player)
    own_moves = len(own_legal_moves)
    for m in own_legal_moves:
        own_moves += len(get_moves(game, m))

    return float(own_moves / opp_moves**2)


def custom_score_2(game, player):
    """Compares computer player legal available moves to a opponent
    player's weighted legal available moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if not opp_moves:
        return float("inf")

    own_moves = len(game.get_legal_moves(player))

    return own_moves / opp_moves * 2.


def custom_score_3(game, player):
    """Calculate the number of legal moves available to the player and negatively
    weight that result if the players current position is in the corner of the
    board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    loc = game.get_player_location(player)
    weight = 1
    corners = [(0, 0), (0, game.height - 1), (game.width - 1, 0),
               (game.width - 1, game.height - 1)]

    # Corner Search
    if loc in corners:
        weight += 1

    return float(len(own_moves)**(1 / weight))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=20.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        moves = game.get_legal_moves(self)
        center_move = round((game.width + 1) /
                            2), round((game.height + 1) / 2)

        if not moves:
            return (-1, -1)
        elif game.move_count <= 2 and game.move_is_legal(center_move):
            # Initialize mvoe to center square if available
            best_move = center_move
        else:
            # Initialize random move if center isn't available
            best_move = moves[0]

        try:
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def min_value(game, depth):
            """ Return the value for a win (+1) if the game is over,
            otherwise return the minimum value over all legal child
            nodes.
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if not game.get_legal_moves() or depth == 0:
                return self.score(game, self)

            v = float("inf")
            for m in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(m), depth - 1))

            return v

        def max_value(game, depth):
            """ Return the value for a loss (-1) if the game is over,
            otherwise return the maximum value over all legal child
            nodes.
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if not game.get_legal_moves() or depth == 0:
                return self.score(game, self)

            v = float("-inf")
            for m in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(m), depth - 1))
            return v

        # Initialize best move
        best_v = float("-inf")
        moves = game.get_legal_moves(self)
        center_move = round((game.width + 1) /
                            2), round((game.height + 1) / 2)

        if not moves:
            return (-1, -1)
        elif game.move_count <= 2 and game.move_is_legal(center_move):
            # Initialize mvoe to center square if available
            best_move = center_move
        else:
            # Initialize random move if center isn't available
            best_move = moves[0]

        for m in game.get_legal_moves():
            v = min_value(game.forecast_move(m), depth - 1)
            if v > best_v:
                best_v = v
                best_move = m

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        moves = game.get_legal_moves(self)
        center_move = round((game.width + 1) /
                            2), round((game.height + 1) / 2)

        if not moves:
            return (-1, -1)
        elif game.move_count <= 2 and game.move_is_legal(center_move):
            # Initialize mvoe to center square if available
            best_move = center_move
        else:
            # Initialize random move if center isn't available
            best_move = moves[0]

        try:
            iter_depth = 1
            while True:
                best_move = self.alphabeta(game, iter_depth)
                iter_depth += 1

        except SearchTimeout:
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def max_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if not bool(game.get_legal_moves()) or depth == 0:
                return self.score(game, self)

            v = float("-inf")
            for m in game.get_legal_moves():
                v = max(v, min_value(
                    game.forecast_move(m), depth - 1, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if not bool(game.get_legal_moves()) or depth == 0:
                return self.score(game, self)

            v = float("inf")
            for m in game.get_legal_moves():
                v = min(v, max_value(
                    game.forecast_move(m), depth - 1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        # Initialize best_move
        best_v = float("-inf")
        moves = game.get_legal_moves(self)
        center_move = round((game.width + 1) /
                            2), round((game.height + 1) / 2)

        if not moves:
            return (-1, -1)
        elif game.move_count <= 2 and game.move_is_legal(center_move):
            # Initialize mvoe to center square if available
            best_move = center_move
        else:
            # Initialize random move if center isn't available
            best_move = moves[0]

        for m in game.get_legal_moves():
            v = min_value(game.forecast_move(m), depth - 1, alpha, beta)
            if v > best_v:
                best_v = v
                best_move = m
            alpha = max(alpha, v)

        return best_move
