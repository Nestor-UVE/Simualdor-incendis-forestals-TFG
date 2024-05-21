import matplotlib.pyplot as plt

# Open the file and read the lines
with open("C:/Users/NESTOR/Documents/UAB/TFG/Repositoris/gym_forestfire/results/Resultats21-5.txt", "r") as f:
    lines = f.readlines()

# Extract the rewards from each line
rewards = []
steps = []
for line in lines:
    try:
        reward = float(line.split(",")[0])
        rewards.append(reward)
        step = int(line.split(",")[1])
        steps.append(step)
    except ValueError:
        continue

# Create a list of episode numbers
episodes = list(range(1, len(rewards) + 1))

#calculate the average steps for last 10 episodes and plot it
average_steps = []
for i in range(100, len(steps)):
    average_steps.append(sum(steps[i-100:i])/100)



#print total average steps
print(sum(average_steps)/len(average_steps))

#make an horizontal line in 33 starting from 0
plt.axhline(y=33, color='r', linestyle='-', label='Random policy')
#vertical line in 50
plt.axvline(x=50, color='g', linestyle='-', label='Random policy off')

# plot average rewards
# plt.plot(list(range(1, len(average_reward)+1)), average_reward)

plt.plot(list(range(1, len(average_steps)+1)), average_steps)

# Plot the rewards
legend = ["Random policy", "Last 100 mean steps", "Random policy off"]
plt.legend(legend)

plt.xlim(0, len(average_steps))
plt.ylim(0, 45)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.show()