#!/usr/bin/python
# -*- coding: latin-1 -*-

from gene import *
from clustering import *
import pandas as pd
from last import *
import math

def get_dis(x, y, x1, y1):
	return math.sqrt((x - x1)**2 + (y - y1)**2)

df = pd.read_csv("city_locations.csv")[:100]
x = np.array(df[['Longitude', 'Latitude']])
starting_pos = 2
group_count = 10
n = 100
y = kmeans(x, group_count) # Culustered list.



Graph = []
for i in range(0, n):
	ls = [0] * n
	Graph.append(ls)

for i in range(0, n):
	for j in range(0, n):
		Graph[i][j] = get_dis(df['Longitude'][i], df['Latitude'][i], df['Longitude'][j], df['Latitude'][j]) 


Model = solve(starting_pos, y, Graph, n)
sol_per_group = Model.get_sol(group_count)

sol_per_group_genetic = Solve(sol_per_group, starting_pos, df[['Longitude', 'Latitude']])
	
