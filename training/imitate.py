import math
import random
import csv
from gym_symmetric.envs.symmetric_env import VehicleEnv, Imitated

env = VehicleEnv()
env.reset()
env.seed(random.randint(0, 1000000))
imitate = Imitated(env)

n = 1000
cumulative_reward = 0
cumulative_square = 0
average_price = 0

act_vehs = []
act_pris = []
filename = 'opt'
with open(f"./static/{filename}_vehicle.txt", 'r') as in_file:
    reader = csv.reader(in_file, delimiter=',')
    for line in reader:
        act_vehs.append(line)
with open(f"./static/{filename}_price.txt", 'r') as in_file:
    reader = csv.reader(in_file, delimiter=',')
    for line in reader:
        act_pris.append(line)
action = [[] for i in range(env.node)]
for i in range(env.node):
    action[i].extend([float(act_vehs[i][j]) / 10 for j in range(env.node)])
    action[i].extend([float(act_pris[i][j]) / 10 for j in range(env.node)])

for cycle in range(n):
    # action = imitate.compute_action()
    _, reward, _, info = env.cycle_step(action)
    reward = reward
    cumulative_reward += reward
    cumulative_square += reward ** 2
    average_price += info["price"]

mean = cumulative_reward / n
var = cumulative_square / n - mean ** 2
print("mean: ", mean, "stdev: ", math.sqrt(var / n))
print("average price", average_price / n)

# Testing commit
