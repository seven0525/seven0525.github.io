---
layout: post
title:  "三目並べの強化学習"
date:   2018-11-29
excerpt: "TensorFlowの練習"
project: true
tag:
- Machine Learning
- TensorFlow
comments: false
---

# NNで三目並べを強化学習させる

## 三目並べをプレイするために、

- マス目の位置
- 9次元のベクトル
- 最善の手 
を与えることで、ニューラルネットワークをトレーニングする。

ここでは、三目並べの限られた数の局面を与えるほか、トレーニングセットのサイズを増やすために各局面にランダムな座標変換を適用する。

このアルゴルズムをテストするために、1つの局面の指し手を削除し、モデルが最善の手を予測できるかどうかを確認する。

最後に実際にモデルと対戦をする。

このレシピの局面と最善の手からなるリストは[ここ](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/08_Learning_Tic_Tac_Toe)に用意されている。

```py
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import random
from tensorflow.python.framework import ops
ops.reset_default_graph()

#モデルをトレーニングするためのバッチサイズを指定(5000を推奨)
batch_size = 50

#局面をXとOで出力する関数
def print_board(board):
    symbols = ['0', ' ','X']
    board_plus1 = [int(x) + 1 for x in board]
    print(' ' + symbols[board_plus1[0]] + ' | ' + symbols[board_plus1[1]] + ' | ' + symbols[board_plus1[2]])
    print('________________')
    print(' ' + symbols[board_plus1[3]] + ' | ' + symbols[board_plus1[4]] +  ' | ' + symbols[board_plus1[5]])
    print('________________')
    print(' ' + symbols[board_plus1[6]] + ' | ' + symbols[board_plus1[7]] +  ' | ' + symbols[board_plus1[8]])

#座標交換を使って新しい局面と最善の手を返す関数
def get_symmetry(board, response, transformation):
    """
    :param board: 長さ9の整数のリスト:
     opposing mark = -1
     friendly mark = 1
     empty space = 0
    :param transformation: 以下5つの座標変換のうちの１つ
     'rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h'
    :return: tuple: (new_board, new_response)
    """
    if transformation == 'rotate180':
        new_response = 8 - response
        return board[::-1], new_response
    elif transformation == 'rotate90':
        new_response = [6, 3, 0, 7, 4, 1, 8, 5, 2].index(response)
        tuple_board = list(zip(*[board[6:9], board[3:6], board[0:3]]))
        return [value for item in tuple_board for value in item], new_response
    elif transformation == 'rotate270':
        new_response = [2, 5, 8, 1, 4, 7, 0, 3, 6].index(response)
        tuple_board = list(zip(*[board[0:3], board[3:6], board[6:9]]))[::-1]
        return [value for item in tuple_board for value in item], new_response
    elif transformation == 'flip_v':
        new_response = [6, 7, 8, 3, 4, 5, 0, 1, 2].index(response)
        return board[6:9] + board[3:6] + board[0:3], new_response
    elif transformation == 'flip_h':  # flip_h = rotate180, then flip_v
        new_response = [2, 1, 0, 5, 4, 3, 8, 7, 6].index(response)
        new_board = board[::-1]
        return new_board[6:9] + new_board[3:6] + new_board[0:3], new_response
    else:
        raise ValueError('Method not implemented.')

#局面と最善の手のリストは、.csvファイルに含まれている。
#このファイルから局面と最善の手を読み込み、タプリのリストとして格納する関数
def get_moves_from_csv(csv_file):
    """
    :param csv_file: 局面と最善の手を含むCSVファイル
    :return: moves: 最善の手のインデックスが含まれた指し手のリスト
    """
    moves = []
    with open(csv_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            moves.append(([int(x) for x in row[0:9]], int(row[9])))
    return moves

#いくつかの関数を組み合わせて、ランダムに変換された局面と最善の手を返す関数
def get_rand_move(moves, rand_transforms=2):
    (board, response) = random.choice(moves)
    possible_transforms = ['rotate90', 'rotate180', 'rotate270', 'flip_v', 'flip_h']
    for _ in range(rand_transforms):
        random_transform = random.choice(possible_transforms)
        (board, response) = get_symmetry(board, response, random_transform)
    return board, response

#グラフセッションを開始し、データロードした後、トレーニングセットを作成する
sess = tf.Session()

#局面と最善の手が含まれたリストを取得
moves = get_moves_from_csv('base_tic_tac_toe_moves.csv')

#トレーニングセットを作成
train_length = 500
train_set = []
for t in range(train_length):
    train_set.append(get_rand_move(moves))

#トレーニングセットから１つの局面の指し手を削除して、このモデルが最善の手を予測できるかどうかを確認したい。
# 次の局面に対する最善の手は、インデックス６のマスに打つことだ。
# To see if the network learns anything new, we will remove
# all instances of the board [-1, 0, 0, 1, -1, -1, 0, 0, 1],
# which the optimal response will be the index '6'.  We will
# Test this at the end.
test_board = [-1, 0, 0, 1, -1, -1, 0, 0, 1]
train_set = [x for x in train_set if x[0] != test_board]

#モデルの変数と演算を作成するための関数を定義する。
#このモデルにソフトマックス活性化関数（softmax()）が含まれていないことに注目しよう。
#この関数は損失関数に含まれている。
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape))


def model(X, A1, A2, bias1, bias2):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, A1), bias1))
    layer2 = tf.add(tf.matmul(layer1, A2), bias2)
    return(layer2)

#プレースホルダ、変数、モデルを設定する
X = tf.placeholder(dtype=tf.float32, shape=[None, 9])
Y = tf.placeholder(dtype=tf.int32, shape=[None])

A1 = init_weights([9, 81])
bias1 = init_weights([81])
A2 = init_weights([81, 9])
bias2 = init_weights([9])

model_output = model(X, A1, A2, bias1, bias2)

#損失関数を設定する。
#トレーニングステップと最適化関数を設定する
#将来このモデルと対戦できるようにしたい場合は、予測関数を作成してく必要もある。
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=Y))
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
prediction = tf.argmax(model_output, 1)

#変数を初期化し、ニューラルネットワークのトレーニングを開始する。
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
for i in range(10000):
    #バッチを選択するためのインデックスをランダムに選択
    rand_indices = np.random.choice(range(len(train_set)), batch_size, replace=False)
    #ランダムな値でバッチを取得
    batch_data = [train_set[i] for i in rand_indices]
    x_input = [x[0] for x in batch_data]
    y_target = np.array([y[1] for y in batch_data])
    #トレーニングステップを実行
    sess.run(train_step, feed_dict={X: x_input, Y: y_target})
    
    #トレーニングセットの損失値を取得
    temp_loss = sess.run(loss, feed_dict={X: x_input, Y: y_target})
    loss_vec.append(temp_loss)
    
    if i % 500 == 0:
        print('Iteration ' + str(i) + ' Loss: ' + str(temp_loss))
```

(こんな感じの結果になる)

Iteration 0 Loss: 8.904177  
Iteration 500 Loss: 1.74531  
Iteration 1000 Loss: 1.4589853  
Iteration 1500 Loss: 1.1635289  
Iteration 2000 Loss: 1.3590469  
Iteration 2500 Loss: 1.3962117  
Iteration 3000 Loss: 0.9921367  
Iteration 3500 Loss: 1.0137455  
Iteration 4000 Loss: 1.0708985  
Iteration 4500 Loss: 1.0389541  
Iteration 5000 Loss: 1.0655681  
Iteration 5500 Loss: 1.1608162  
Iteration 6000 Loss: 0.8579481  
Iteration 6500 Loss: 0.9746061  
Iteration 7000 Loss: 0.7223426  
Iteration 7500 Loss: 0.7803637  
Iteration 8000 Loss: 0.8147448  
Iteration 8500 Loss: 0.7135554  
Iteration 9000 Loss: 0.6399172  
Iteration 9500 Loss: 0.7355053  

## このモデルのトレーニングセットの損失値をプロットする。

```py
plt.plot(loss_vec, 'k-', label='Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
```

![aaa](data:image/png;base64,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iOOOO46dd945280REWkyy/RvvZrZzsBkYIi7v9BQ3d69e/vUqVObZbnz589n3333bZZ5ASxcuJCioiKOPPJIZs+ezYEHtljvlIhISmZW6O6906mb0T18M2sHPA881VjYN7eePXsyYsSIZpvf3nvvzVFHHcVee+3FQQcdxLhx4zj33HMpLy+P1ykrK6OoqAgIfkdXP5wuIrkkk2fpGPA4MNvd783UchpyySWXsP322zfrPGOB3r9/f8aMGZP0oXLsscey5557MmPGDAoKChg6dCgbNmxo1uWLiGytjHXpmNnPgfeBr4CacPBgd3891TTN2aWTaMqUKfz85z9v9vnGVFdXs80228RvzXzOOefw9NNPx8drT19EMmVLunTaZqoR7v4B0PDN6VvIkUcemdH5Dxo0iOuvvz7+PDHsAUpKSrjqqquoqqqiZ8+enHPOOQ1eCVxTU8OmTZvYYYcdMtZmEYmejB+03RKZ2sMHmDhxIscdd1xG5r2lunbtyrJlywA44YQTeOutt5K+BQwePJihQ4dSWlrKTjvtlK1mikgrkDMHbXPJscceG+9/z7bly5czduxYNm3axFtvvRUfXl1dTa9eveI3f1u/fj0Q7PGPHTuW4cOHU1ZWRlVVFT/+8Y957bXX6sz7ww8/5NFHH+Whhx7CzFi1alV83tm+UdzixYv128Ii2eTuOVMOP/xwz7Qnn3zSgZwoV1xxRfzx6aefXmf81Vdf7aWlpf7www/Hhw0cONCXLVsWf57o7rvvjg/v0KGDA15YWOju7jfddJMDPmfOnIy/xql06dKlTptFpGmAqZ5mxmY95BNLSwR+SUmJ9+nTJ+thn245++yz/c9//nPSsJNOOin++LvvvnN3948//rje6WOBv/322zvgqV7jkpKSrXo9lyxZ4hs2bEirbn0fUiLSNAr8NMyePTvrYd5c5YQTTkg5rl27dr5w4cKkYZWVlUmvxVdffeUQfOMoKSnxV1991QEvLi6O13njjTd83bp1dV5HwA855JB6X+OSkhK///77ffbs2fG6iYE/adIk/+yzz5pjc4pElgI/TdkO6myV888/32tqanzChAleWVnpY8aMSRof+wZ0/fXXe01NTfzbQ79+/fzZZ5+t9zWsT48ePZLG167b0LQikh4FfpoAb9u2bVLY/eEPf8h6IOd6GTJkiFdWVvqaNWuSQvv555/3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%0A)

## テスト

```py
#テスト
#トレーニングセットから削除した局面でテストを実行したらどうなるかを確認する
#[test_bord]を予測させてみる（最善の手はインデックス6）
test_boards = [test_board]
feed_dict = {X: test_boards}
logits = sess.run(model_output, feed_dict=feed_dict)
predictions = sess.run(prediction, feed_dict=feed_dict)
print(predictions)

#評価
#トレーニングしたモデルと対戦する計画を立てる
#勝敗をチェックする関数を作成する必要がある。
def check(board):
    wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for ix in range(len(wins)):
        if board[wins[ix][0]] == board[wins[ix][1]] == board[wins[ix][2]] == 1.:
            return 1
        elif board[wins[ix][0]] == board[wins[ix][1]] == board[wins[ix][2]] == -1.:
            return 1
    return 0
    
#評価
#トレーニングしたモデルと対戦する計画を立てる
#勝敗をチェックする関数を作成する必要がある。
def check(board):
    wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for ix in range(len(wins)):
        if board[wins[ix][0]] == board[wins[ix][1]] == board[wins[ix][2]] == 1.:
            return 1
        elif board[wins[ix][0]] == board[wins[ix][1]] == board[wins[ix][2]] == -1.:
            return 1
    return 0
    
#最初は、全てのマス目が空（0）
#次に、プレイヤーがインデックスを入力し、そのインデックスをモデルに羽根井して次の一手を予測させる
game_tracker = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
win_logical = False
num_moves = 0
while not win_logical:
    player_index = input('Input index of your move (0-8): ')
    num_moves += 1
    # Add player move to game
    game_tracker[int(player_index)] = 1.
    
    # Get model's move by first getting all the logits for each index
    [potential_moves] = sess.run(model_output, feed_dict={X: [game_tracker]})
    # Now find allowed moves (where game tracker values = 0.0)
    allowed_moves = [ix for ix, x in enumerate(game_tracker) if x == 0.0]
    # Find best move by taking argmax of logits if they are in allowed moves
    model_move = np.argmax([x if ix in allowed_moves else -999.0 for ix, x in enumerate(potential_moves)])
    
    # Add model move to game
    game_tracker[int(model_move)] = -1.
    print('Model has moved')
    print_board(game_tracker)
    # Now check for win or too many moves
    if check(game_tracker) == 1 or num_moves >= 5:
        print('Game Over!')
        win_logical = True

```

