import numpy as np
import matplotlib.pyplot as plt

# Problem setup
win_amount = 10000
value_arr = np.zeros(win_amount+1)
value_arr[win_amount] = 1  # Goal state value
theta = 0.01  # Convergence threshold
p_head = 0.4
reward = np.zeros(win_amount+1)  # Reward for each state
#reward[win_amount] = 1  # Reward only for reaching the goal
opt_bets = np.zeros(win_amount+1)  # Optimal bet for each state

# Value Iteration
sweep_count = 0
while True:
    delta = 0
    sweep_count += 1  # Increment sweep counter
    for s in range(1, win_amount):  # Exclude terminal states
        v = value_arr[s]
        max_val = 0
        opt_bet = 0
        for bet_amount in range(1, min(s, win_amount - s) + 1):  # Possible bet amounts
            # Calculate expected value for head and tail outcomes
            win_state = min(s + bet_amount, win_amount)
            lose_state = max(s - bet_amount, 0)
            value_head = p_head * (reward[win_state] + value_arr[win_state])
            value_tail = (1 - p_head) * (reward[lose_state] + value_arr[lose_state])
            curr_val = value_head + value_tail
            if curr_val > max_val:
                max_val = curr_val
                opt_bet = bet_amount
        delta = max(delta, abs(v - max_val))
        value_arr[s] = max_val
        opt_bets[s] = opt_bet
    if delta < theta:
        break

# Plotting the value function and optimal bets in separate graphs but the same window
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

# Plot Value Function
ax1.plot(value_arr, label='Value Function', color='tab:red')
ax1.set_title('Value Function vs State')
ax1.set_xlabel('State')
ax1.set_ylabel('Value Estimates')
ax1.grid(True)
ax1.legend()

# Plot Optimal Bet Amount
ax2.plot(opt_bets, label='Optimal Bet Amount', color='tab:blue')
ax2.set_title('Optimal Bet Amount vs State')
ax2.set_xlabel('State')
ax2.set_ylabel('Optimal Bet Amount')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()


sweep_count
