import numpy as np
from matplotlib import pyplot as plt

fname = 'TY_climate_2015_2018.csv'
f = open(fname, encoding='cp950')
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
del lines[0]

raw_data = []
for i, line in enumerate(lines):
    value = float(line.split(',')[8])
    raw_data.append([value])

raw_data = np.array(raw_data)
plt.plot(raw_data)
plt.show()

plt.plot(raw_data[:720])
plt.show()