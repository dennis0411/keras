from keras.utils import np_utils

orig = [4, 6, 8]
NUM_DIGITS = 15

for o in orig:
    print("orig={}, shift={}".format(o, np_utils.to_categorical(o, NUM_DIGITS)))
