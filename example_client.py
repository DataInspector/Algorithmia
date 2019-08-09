# Copyright 2019 H2O.ai; Proprietary License;  -*- encoding: utf-8 -*-

import sys

sys.path.append('gen-py')

from numpy import nan
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from h2oai_scoring import ScoringService
from h2oai_scoring.ttypes import Row

# ------------------------------------------------------------
# Name        Type      Range                                 
# ------------------------------------------------------------
# sepal_len   float32   [4.400000095367432, 7.699999809265137]
# sepal_wid   float32   [2.0, 4.199999809265137]              
# petal_len   float32   [1.0, 6.699999809265137]              
# petal_wid   float32   [0.10000000149011612, 2.5]            
# ------------------------------------------------------------

socket = TSocket.TSocket('localhost', 9090)
transport = TTransport.TBufferedTransport(socket)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = ScoringService.Client(protocol)
transport.open()

server_hash = client.get_hash()
print('Scoring server hash: '.format(server_hash))

print('Scoring individual rows...')
row1 = Row()
row1.sepalLen = '5.199999809265137'  # sepal_len
row1.sepalWid = '2.0'  # sepal_wid
row1.petalLen = '1.2999999523162842'  # petal_len
row1.petalWid = '1.5'  # petal_wid

row2 = Row()
row2.sepalLen = '4.5'  # sepal_len
row2.sepalWid = '2.0'  # sepal_wid
row2.petalLen = '1.2000000476837158'  # petal_len
row2.petalWid = '0.20000000298023224'  # petal_wid

row3 = Row()
row3.sepalLen = '4.400000095367432'  # sepal_len
row3.sepalWid = '2.0'  # sepal_wid
row3.petalLen = '1.0'  # petal_len
row3.petalWid = '1.399999976158142'  # petal_wid

row4 = Row()
row4.sepalLen = '4.5'  # sepal_len
row4.sepalWid = '2.5999999046325684'  # sepal_wid
row4.petalLen = '3.299999952316284'  # petal_len
row4.petalWid = '1.5'  # petal_wid

row5 = Row()
row5.sepalLen = '5.400000095367432'  # sepal_len
row5.sepalWid = '3.0'  # sepal_wid
row5.petalLen = '1.899999976158142'  # petal_len
row5.petalWid = '0.10000000149011612'  # petal_wid

score1 = client.score(row1, pred_contribs=False, output_margin=False)
print(score1)
score2 = client.score(row2, pred_contribs=False, output_margin=False)
print(score2)
score3 = client.score(row3, pred_contribs=False, output_margin=False)
print(score3)
score4 = client.score(row4, pred_contribs=False, output_margin=False)
print(score4)
score5 = client.score(row5, pred_contribs=False, output_margin=False)
print(score5)

print('Scoring multiple rows...')
rows = [row1, row2, row3, row4, row5]
scores = client.score_batch(rows, pred_contribs=False, output_margin=False)
print(scores)

print('Retrieve column names')
print(client.get_column_names())

print('Retrieve transformed column names')
print(client.get_transformed_column_names())

print('Retrieve target labels') # will be not empty only for classification problems
print(client.get_target_labels())

transport.close()


