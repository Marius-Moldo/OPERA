import numpy as np

path = "datasets/covidUK/"
# path = ''

filenames = np.load(path + "entire_" + "cough" + "_filenames.npy")

print(filenames[0])
