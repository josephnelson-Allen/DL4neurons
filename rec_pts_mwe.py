import logging as log

from run import get_model
from get_rec_points import get_rec_points

# model = get_model('BBP', log, 'L4_BTC', 'cNAC', 0)
model = get_model('BBP', log, 'L23_LBC', 'bAC', 0)

x = get_rec_points(model.entire_cell)
y = get_rec_points(model.entire_cell)

print("First time: {}".format(x))
print("Second time: {}".format(y))

print("First time: {} rec points".format(len(x)))
print("Second time: {} rec points".format(len(y)))

