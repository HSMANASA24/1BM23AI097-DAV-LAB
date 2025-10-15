#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

games = np.array([1, 2, 3, 4, 5, 6])
points_scored = np.array([95, 102, 78, 88, 110, 100])
opponent_points = np.array([90, 88, 100, 85, 95, 100])
attendance = np.array([1500, 1800, 1600, 2000, 1900])
change_in_points = np.diff(points_scored, prepend=points_scored[0])

print("Game\tPoints Scored\tChange from Previous Game")
for g, p, c in zip(games, points_scored, change_in_points):
    print(f"[g]\t{p}\t\t{c:+}")


plt.figure(figsize=(8, 5))
plt.plot(games, points_scored, marker='o', color='b', label='points scored')
plt.title("Team's points Scored over the Season")
plt.xlabel("Game Number")
plt.ylabel("Points Scored")
plt.grid(True)
plt.legend()
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
games = np.array([1, 2, 3, 4, 5, 6])
attendance = np.array([1500, 1800, 1600, 2000, 1900])
average_attendance = np.mean(attendance)
print(f"Avg attendance over the season {average_attendance:.2f}")
plt.figure(figsize=(8, 6))
plt.plot(games, points_scored, marker='o', color='b', label='points scored')
plt.axhline(y = average_attendance, color='red', linestyle='--', label=f'Average Attendance')

plt.title("Attendance per Game Over the Season")
plt.xlabel("Game Number")
plt.ylabel("Attendance")
plt.xticks(games)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt

players = ['A', 'B', 'C', 'D', 'E']
points = np.array([
    [20,15,25,18,17],
    [22,18,20,10,13],
    [15,12,28,10,13],
    [18,20,22,15,13],
    [25,25,30,20,10],
    [20,22,18,30,10]
])

total_points = np.sum(points, axis=0)

top_player_index = np.argmax(total_points)
top_player =  players[top_player_index]
top_score = total_points[top_player_index]

print("Total points scored by beachother:")
for p,s in zip(players, total_points):
    print(f"{p}:{s}")

print(f"\n Player {top_player} scored the most points:{top_score}")

plt.figure(figsize=(8,5))
plt.bar(players, total_points, color='skyblue')
plt.title("Total Points Scored by Each Player")
plt.xlabel("Player")
plt.ylabel("Total Points Scored")
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.bar(top_player, top_score, color='orange', label=f'Top Scorer:{top_player} ({top_score} pts)')
plt.legend()
plt.tight_layout()
plt.show()


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {
    "games": [1, 2, 3, 4, 5, 6],
    "points_scored": [95, 102, 78, 88, 110, 100],
    "opponent_points":[90, 88, 100, 85, 95, 100],
    "attendance": [1500, 1800, 1200, 1600, 2000, 1900]
} 
points_scored = [95, 102, 78, 88, 110, 100]
threshold = 100
count_above_threshold = sum(1 for points in points_scored if points > threshold)
print(f"number of games more than {threshold} points: {count_above_threshold}")

df = pd.DataFrame(data)

bins = np.array([70, 80, 90, 100, 110, 120])
labels = ["70-80", "80-90", "90-100", "100-110", "110-120"]

bin_indices = np.digitize(df["points_scored"], bins, right=False)
counts = np.bincount(bin_indices, minlength=len(bins)+1)[1:len(labels)+1]



plt.bar(labels, counts, color="lightgreen",edgecolor="black")
plt.title("Number of Games Scored in Different Ranges (Using NumPy)")
plt.xlabel("Points Scored Range")
plt.ylabel("Number of Games")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt

games = np.array([1, 2, 3, 4, 5, 6])
opponents = np.array(['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F'])
points_scored = np.array([95, 102, 78, 88, 110, 100])
opponent_points = np.array([90, 88, 100, 85, 95, 100])

points_diff = points_scored - opponent_points
best_index = np.argmax(points_diff)
best_opponent = opponents[best_index]
best_diff = points_diff[best_index]

print("Points scored vs opponents:")
for o, ps, op, diff in zip(opponents, points_scored, opponent_points, points_diff):
    print(f"{o}: Scored {ps}, Opponent {op}, Difference {diff:+}")

print(f"\nBest performance: Against {best_opponent} (won by {best_diff} points)")

plt.figure(figsize=(8, 5))
bars = plt.bar(opponents, points_scored, color='skyblue', edgecolor='black')
bars[best_index].set_color('orange')

plt.title("Team's Points Scored Against Each Opponent")
plt.xlabel("Opponent")
plt.ylabel("Points Scored")
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.legend([bars[best_index]], [f'Best vs {best_opponent} (+{best_diff})'])
plt.tight_layout()
plt.show()




# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {
    "Game": [1, 2, 3, 4, 5, 6],
    "PointsScored": [95, 102, 78, 88, 110, 100],
    "OpponentPoints": [90, 88, 100, 85, 95, 100],
    "Attendance": [1500, 1800, 1200, 1600, 2000, 1900],
    "Opponent": ["A", "B", "C", "D", "E", "F"]
}
df = pd.DataFrame(data)

df_sorted = df.sort_values(by="Attendance", ascending=False)

plt.bar(df_sorted["Opponent"], df_sorted["Attendance"], color="orange", edgecolor="black")
plt.title("Game Attendance for Different Opponents")
plt.xlabel("Opponent")
plt.ylabel("Attendance")
plt.grid(axis="y", linestyle="--", alpha=0.7)

for i, v in enumerate(df_sorted["Attendance"]):
    plt.text(i, v + 30, str(v), ha='center', fontweight='bold')

plt.show()




# In[ ]:





# In[ ]:




