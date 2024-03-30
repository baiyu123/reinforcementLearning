import numpy as np

class wumpus_world:
    def __init__(self):
        self.map = [['e', 'e', 'g', 'p'],
                    ['p', 'e', 'p', 'e'],
                    ['e', 'e', 'e', 'e'],
                    ['e', 'e', 'p', 'e']]
        self.opt_A = np.zeros((4, 4))
        self.Q_values = np.zeros((4, 4, 4))  # States (x,y) by Actions (up, right, down, left)
        self.e = 0.1
        self.discount = 0.8
        self.actions_list = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        self.x, self.y, self.a = 4, 4, 4  # dimensions for policies array

        # Initialize policies with uniform probability distribution over actions
        self.policies = np.full((self.x, self.y, self.a), 1/self.a)  # Each action has an equal probability of 1/a
        self.returns_table = {}

    def generate_episode(self):
        loc = np.array([3, 0])
        episode = []
        final_reward = 0
        while self.map[loc[0]][loc[1]] == 'e':
            action_probabilities = self.policies[loc[0], loc[1], :]
            action = np.random.choice(len(action_probabilities), p=action_probabilities)
            selected_action = self.actions_list[action]
            new_loc = loc + selected_action
            
            if not (0 <= new_loc[0] < self.x and 0 <= new_loc[1] < self.y):
                continue
            
            episode.append([loc.tolist(), action])
            loc = new_loc  # Update location after validation

            if self.map[loc[0]][loc[1]] == 'g':
                final_reward = 100
                break
            elif self.map[loc[0]][loc[1]] == 'p':
                final_reward = -100
                break
        
        return episode, final_reward
    
    def update_returns(self, state, action, return_value):
        # Create a key as a tuple of state and action
        key = (tuple(state), action)
        
        # Check if the key exists in the dictionary
        if key not in self.returns_table:
            # If not, initialize the list for this key
            self.returns_table[key] = []
        
        # Append the new return value to the list for this state-action pair
        self.returns_table[key].append(return_value)

    def get_average_return_for_pair(self, state, action):
        key = (tuple(state), action)
        if key in self.returns_table and len(self.returns_table[key]) > 0:
            return sum(self.returns_table[key]) / len(self.returns_table[key])
        else:
            return 0 

    
    def epi_soft(self):
        episode, final_reward = self.generate_episode()
        G = 0
        for i in range(len(episode)-1, -1, -1):
            pos, action = episode[i]
            R = -1
            if i == len(episode)-1:
                R = final_reward
            G = self.discount*G + R
            self.update_returns(pos, action, G)
            opt_act = 0
            max_q = -1000
            for j in range(0, 4):
                Q = self.get_average_return_for_pair(pos, j)
                if Q > max_q:
                    opt_act = j
                    max_q = Q
            for j in range(0, 4):
                if j == opt_act:
                    self.policies[pos[0]][pos[1]][j] = 1 - self.e + self.e/4
                else:
                    self.policies[pos[0]][pos[1]][j] = self.e/4
            


wp = wumpus_world()
for i in range(0, 100):
    wp.epi_soft()
print(wp.policies)