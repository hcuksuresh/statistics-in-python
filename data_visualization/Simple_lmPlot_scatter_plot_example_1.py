# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:52:54 2017

@author: sarwesh suman
"""

import pandas as pd
import seaborn as sns
import os

file_name=r'C:\Users\bgh47373\Documents\Module -2\part1\classification\binary.csv'

df = pd.read_csv(file_name)

print(df.head())

""" plotting gre vs gpa """

sns.lmplot(x='gre',y='gpa',data=df)