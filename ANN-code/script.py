#!/usr/bin/env python3
"""
SSH SCRIPT
"""

import pandas as pd
import os

df0 = pd.read_csv('features_0.csv')
df1 = pd.read_csv('features_1.csv')
df2 = pd.read_csv('features_2.csv')
df3 = pd.read_csv('features_3.csv')
df4 = pd.read_csv('features_4.csv')
df5 = pd.read_csv('features_5.csv')
df6 = pd.read_csv('features_6.csv')
df7 = pd.read_csv('features_7.csv')
df8 = pd.read_csv('features_8.csv')
df9 = pd.read_csv('features_9.csv')

df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9])

df.to_csv('features_CF4.csv', index=False)