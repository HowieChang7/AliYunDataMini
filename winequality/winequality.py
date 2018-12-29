# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

red = pd.read_csv("winequality-red.csv",sep=";")
# print(red.head(10))
# print("red.info:",red.info())
# print("red.describe:",red.describe())

plt.style.use("ggplot")
conlum = red.columns.tolist()
plt.figure()
