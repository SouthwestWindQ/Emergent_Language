import json
import random


init_states = [''.join((str(i), str(j), str(k))) for i in range(3) for j in range(3) for k in range(3)]
goal_states = [''.join((str(i), str(j), str(k))) for i in range(3) for j in range(3) for k in range(3)]

random.seed(0)
random.shuffle(goal_states)
    
with open('rule.json', 'w') as file:
    json.dump({init_state: goal_state for init_state, goal_state in zip(init_states, goal_states)}, file)
