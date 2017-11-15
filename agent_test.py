"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest
import timeit
from copy import copy

from isolation import Board
from sample_players import RandomPlayer
from sample_players import GreedyPlayer
import game_agent

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer()
        self.player2 = game_agent.MinimaxPlayer()
        self.game = Board(self.player1, self.player2)

    def setUpAB(self, search_depth=3):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer(search_depth)
        self.player2 = game_agent.AlphaBetaPlayer(search_depth)
        self.game = Board(self.player1, self.player2)

    def test_v_ab(self):
        self.setUpAB()
        # time_left = lambda: 99
        # self.game._active_player.get_move(self.game, time_left)
        self.game.play()

    def test_v_mm(self):
        self.setUp()
        # time_left = lambda: 99
        # self.game._active_player.get_move(self.game, time_left)
        self.game.play()


if __name__ == '__main__':
    unittest.main()
