# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:03:51 2020

@author: Niranch
"""
#TicTactoe Game - Using random module, code to make random postions in tic tac board for player 1 and player 2,
#verify if there is a winner and get the number of times player 1, player 2 won, and 
#number of times game was a draw.

import numpy as np
import random

#Initialize board with all zeros
def create_board():
    board = np.zeros ((3,3), dtype=int)
    return board

#board = create_board()

#Make a position for the player if position is not taken (ie. position == 0)
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
    return board
#place(board, 1, (0, 0))

#Use Zip function to send the iterator list to find the list of unoccupied position in board
def possibilities(board):
    return list(zip(*np.where(board == 0)))

#random.seed(1)
#If board is free to make a selection then use random.choice to select a random position 
#sent by possibilities method
def random_place(board, player):
    selections = possibilities(board)
    if len(selections) > 0:
        selection = random.choice(selections)
        place(board, player, selection)
    return board

#random_place(board, 2)


#for i in range(3):
#    for player in [1, 2]:
#        random_place(board, player)
#print(board)

#If any of the row has all the elements placed by given player then declare winner
def row_win(board, player):
    if np.any(np.all(board == player, axis=1)):
        return True
    else:
        return False


#If any of the col has all the elements placed by given player then declare winner
def col_win(board, player):
    if np.any(np.all(board == player, axis=0)):
        return True
    else:
        return False



#board[1,1] = 2
#Check diagonal postion left to right and the reverse by using flipr method
def diag_win(board, player):
    if np.all(np.diag(board)==player) or np.all(np.diag(np.fliplr(board))==player):
        return True
    else:
        return False


#Call appropriate methods to check for winner 
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player

    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner


# Running Tic-tac-toe
#Begin the game by creating board, finding possible pos's, placing player and evaluating winner
def Play_game():
    board = create_board()
    possibilities(board)
#    board[1,1] = 1
    winner = 0
    while winner == 0:
        for player in [1,2]: 
            random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break

#    print(board)
#    print(winner)
    return winner


#Play_game()
#Play Game for 1000 times and get the result every time in results object
results = [Play_game() for i in range(1000)]
#Print the times Player 1 won the game
print ("Player 1 won ",results.count(1)," times")
#Print the times Player 2 won the game
print ("Player 2 won ",results.count(2)," times")
#Print the times game is draw
print ("Game draw ",results.count(-1)," times")