# AI Project: Landing On The Moon

# 1-Overview:

Welcome to the Landing On The Moon repository! This project is designed to train an AI Agent to learn how to land between two poles on the moon using Deep Q-learning on the  gymnasium box2d enviroment.

# 2-Tools:

1-pytorch

2-gymnasium

# 3-training:

1-The training process runs for a speciifed number of episodes in my case i choose 2000

2-At the beginning of each episode, the environment is reset, and the initial state is obtained.

3-Each episode consists of a series of time steps.

4-The agent selects an action based on the current state and the epsilon-greedy policy.

5-The chosen action is executed in the environment, resulting in the next state, reward, and a done flag indicating whether the episode has terminated.

6-The state is updated to the next state, the cumulative reward (score) is updated, and if the episode is done, the loop breaks.

7-Epsilon is decayed after each episode to gradually reduce the exploration rate.

# 4-Performance Monitoring:

1-The cumulative score for the episode is appended to the scores deque.

2-The average score over the last 100 episodes is printed to monitor progress.

3-If the average score over the last 100 episodes reaches 200.0, the environment is considered solved. The model's state dictionary is saved to 'checkpoint.pth', and training is stopped.

# 5-Learning Process
## i)Experience Replay:
The agent uses experience replay to store past experiences in replay memory. Periodically, a batch of experiences is sampled to update the Q-network. This helps stabilize training by breaking the correlation between consecutive experiences.

## ii)Target Network:
The target Q-network provides stable target values for the Q-learning update. The target network is slowly updated towards the local Q-network using a soft update mechanism controlled by interpolation_parameter_tau variable.

## iii)Q-Learning Update
The agent updates its Q-values using the Bellman equation:

𝑄
(
𝑠
,
𝑎
)
←
𝑄
(
𝑠
,
𝑎
)
+
𝛼
[
𝑟
+
𝛾
max
𝑎
′
𝑄
′
(
𝑠
′
,
𝑎
′
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)←Q(s,a)+α[r+γmax 
a 
′
 Q 
′
 (s 
′
 ,a 
′
)−Q(s,a)]

where 
𝑄
′
Q 
′
  represents the target Q-network and 
𝛼
α is the learning rate. The loss is computed using mean squared error between the expected and target Q-values, and the network is trained to minimize this loss.





