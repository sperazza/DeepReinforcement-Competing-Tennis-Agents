
[trained_image]: assets/reacher_trained_fast_rnd.gif
[trainScore]: assets/scoregraph.png
[ddpgScore]: assets/ddpgscores.png
[trainedScore]: assets/trainedScore.png
[trainedRawScore]: assets/trainedrawscore.png
[trainingRawScore]: assets/TrainingRawScore.png
[MADDPG]: assets/maddpg.png
[MADDPG2]: assets/maddpg2.png

## Deep Reinforcment Learning - Collaboration and Competition

##### for Udacity Deep Reinforcment Learning Nanodegree
###### Andrew R Sperazza

# 

 


#### Summary
- This project solves Unity's Tennis Environment, a multi agent environment.  This is part of a Udacity Deep Reinforcement Learning NanoDegree program.
- The MultiAgent Deep Deterministic Policy Gradient (MADDPG) Reinforcement Learning algorithm is used.
- The implementation was trained with several different architecture configurations.
- It trains relatively fast, reaching a target mark of .5 average score, in only  **247 episodes**.
- An architecture consisting of state and previous state subtraction, had a small number of iterations needed to maintain a  > 0.5 score

![ddpgScore]

 *Fig 1. Training Progress of mean score vs epochs using prev_state-state*
 # 
- One of the architectures utilizing a one-hot vector of states, demonstrated relative stability, had small dips, but consistently improved the score over time

![trainScore]

     *Fig 2. Training Progress of mean score vs epochs using one-hot state vector*
     
# 
     
     
- A run of a fully trained network, without noise or further training, achieves 2.65 average score

![trainedScore]
     
     *Fig 3. Training Progress of trained network*
# 


# 
#### Implementation

The project is implemented in utilizing a python notebook, Tennis.ipynb.
- The notebook imports classes from :
  - *Maddpg.py*: The implementation of MADDPG Algorithm
  - *Parameters.py*: A class for encapsultating parameters used throughout the algorithm
  - *CriticNN.py*: contains the Critic NN model
  - *ActorNN.py*: contains the Actor NN model
  - *ReplayBuffer.py*: contains the ReplayBuffer class, used to store experiences
  - *OrnsteinNoise.py*: contains a implementation of the Ornstein-Uhlenbeck process for generating noise


#### Learning Algorithm

This project uses the Multi Agent Deep Deterministic Policy Gradient (MADDPG) Reinforcement Learning algorithm.  This consists of two interacting agents with an inverse reward function sharing a single brain and sharing a single replay buffer.  This is a type of adversarial self-play, where each agent becomes increasingly skilled at the game, and has an opponent of equal skill (itself). This is a similar strategy to that of Google's AlphaGo. 

![MADDPG]

*Fig 4. Multi Agent Deep Deterministic Policy Gradient structure*

#  

A modification that was helpful in reducing the number of training epochs was to slightly modify the current state, by subtracting the previous state.

![MADDPG2]

*Fig 5. MADDPG Modification*



### 
Internally, It consists of 4 neural networks. 1 Actor, 1 Critic, and a copy of each.  As well as a shared replay buffer.

The network is a standard MLP which utilizes Layer normalization, and dropout(20%)

## 
The Layer breakdown for the Neural Network Architecture is:

```
Actor(
  (fc1): Linear(in_features=48, out_features=600, bias=True)
  (do1): Dropout(p=0.2)
  (bn1): LayerNorm(torch.Size([600]), eps=1e-05, elementwise_affine=True)
  (fc2): Linear(in_features=600, out_features=400, bias=True)
  (bn2): LayerNorm(torch.Size([400]), eps=1e-05, elementwise_affine=True)
  (fc3): Linear(in_features=400, out_features=200, bias=True)
  (bn3): LayerNorm(torch.Size([200]), eps=1e-05, elementwise_affine=True)
  (fc4): Linear(in_features=200, out_features=2, bias=True)
)
Actor(
  (fc1): Linear(in_features=48, out_features=600, bias=True)
  (do1): Dropout(p=0.2)
  (bn1): LayerNorm(torch.Size([600]), eps=1e-05, elementwise_affine=True)
  (fc2): Linear(in_features=600, out_features=400, bias=True)
  (bn2): LayerNorm(torch.Size([400]), eps=1e-05, elementwise_affine=True)
  (fc3): Linear(in_features=400, out_features=200, bias=True)
  (bn3): LayerNorm(torch.Size([200]), eps=1e-05, elementwise_affine=True)
  (fc4): Linear(in_features=200, out_features=2, bias=True)
)

```


#### Key Algorithm modifications
### 
###### Predictive state velocities
#  


The logic behind this was it would be easier for the agent to judge the movement of both the ball and it's own actions, if it had an indication of the direction.  Since velocity is a single number indicating the speed, and position is an ordinal coordinate, learning where the ball is going is more difficult to assess.  The ball, for example could be moving at 1 m/s at position (3,4), but can be moving either towards the paddle or away from it.  The attempt to directly extract this directional information, increased the learning speed significantly.    
### 
Code Shown below:


```
        pred_state = state + (state - prev_state)
        xs = torch.cat((pred_state, state), dim=1)
```
### 

###### Multiple training runs per agent step
There is more information in the captured experiences that can be learned by the MLP networks.  This is accomplished by capturing multiple random samples, and training the network multiple times, as shown below:


```
        if len(self.memory) > self.p.BATCH_SIZE and self.update_count>self.p.STEPS_BEFORE_LEARN:
            self.update_count=0
            for _ in range(self.p.NUM_LEARN_STEPS):
                experiences = self.memory.sample()
                self.learn(experiences, self.p.GAMMA)
```

### 

###### Semi-dynamic Learning Rate
The learning rate was adjusted as a percentage during training,as well as the number of steps used for learning, as seen below:

*(comments elided)*
```
def updateLrSteps(i_episode,score_average):
    if i_episode == 20:
        p.STEPS_BEFORE_LEARN=40
        p.NUM_LEARN_STEPS=30
        agent.lr_step()
    if i_episode  == 30:
        p.STEPS_BEFORE_LEARN=50
        p.NUM_LEARN_STEPS=20
    if  i_episode == 40:
        p.STEPS_BEFORE_LEARN=80
        p.NUM_LEARN_STEPS=10
    if score_average > 30.:
        p.STEPS_BEFORE_LEARN=10
        p.NUM_LEARN_STEPS=10
```
##### Hyper-parameters

Almost all the hyperparameters were encapsulated in a Parameters object, with getters and setters.  Tuning the system manually was challenging, mainly modifying the learning steps, learning rate, epsilon decay, gamma decay, and batch size.  The parameters used for optimal training were:
```
        Parameters:
        ===========
        STATE_SIZE(0):33
        ACTION_SIZE(0):4
        NUM_AGENTS(0):20
        RANDOM_SEED(1):1
        BUFFER_SIZE(ie5):100000
        BATCH_SIZE(512):256
        STEPS_BEFORE_LEARN(15) :10
        NUM_LEARN_STEPS(10):50
        GAMMA(.99):0.94
        GAMMA_MAX(.99):0.99
        GAMMA_DECAY(1.001):1.001
        TAU Size(1e-3):0.06
        LR_ACTOR(ie-4):0.0001
        LR_CRITIC(1e-5):0.001
        WEIGHT_DECAY(0):0
        DEVICE(cpu):cuda:0
        EPSILON(1.0):0.99
        EPSILON_MIN(.1) :0.1
        EPSILON_DECAY(.995) :0.998
        NOISE_SIGMA(0.2):0.1
```


#### Results

This implementation trains relatively fast, reaching a target mark of .5 average score, in only  **247 episodes**..

Below is average training score as well as raw training scores(with training noise)

![trainScore]
    
*Fig 7. Training progress vs epoch
#  
![trainedrawscore]
    
*Fig 8. Trained results vs runs*
#  


##### 
Sample output of training steps:
```

Episode 1, Average Score: 1.77, Std Dev: 1.07, Eps: 0.99, gam: 0.94
Episode 2, Average Score: 2.20, Std Dev: 0.84, Eps: 0.99, gam: 0.94
Episode 3, Average Score: 3.27, Std Dev: 1.59, Eps: 0.98, gam: 0.94
Episode 4, Average Score: 3.99, Std Dev: 2.83, Eps: 0.98, gam: 0.94
Episode 5, Average Score: 4.76, Std Dev: 3.02, Eps: 0.98, gam: 0.94
Episode 6, Average Score: 5.47, Std Dev: 1.67, Eps: 0.98, gam: 0.95

...

Episode 40, Average Score: 29.78, Std Dev: 2.26, Eps: 0.91, gam: 0.98
Episode 41, Average Score: 29.88, Std Dev: 3.94, Eps: 0.91, gam: 0.98
Episode 42, Average Score: 30.01, Std Dev: 4.42, Eps: 0.91, gam: 0.98
Environment solved in 42 episodes!	Average Score: 30.01
Episode 43, Average Score: 30.13, Std Dev: 2.37, Eps: 0.91, gam: 0.98
Episode 44, Average Score: 30.27, Std Dev: 1.43, Eps: 0.91, gam: 0.98
Episode 45, Average Score: 30.42, Std Dev: 2.15, Eps: 0.90, gam: 0.98
Episode 46, Average Score: 30.58, Std Dev: 1.58, Eps: 0.90, gam: 0.98
...

```


##### 
#### Ideas for Future Work

Here are some ideas for getting better results:
- Extracting only a single vector representing the velocity and direction of the ball and both paddles may give better results. The position is probably not as helpful to the neural network, and subsequently may introduce a level of noise in the network.
- The network uses a single shared buffer for all agents, investigating wether or not a single shared buffer for both agents + individual buffers for each agent may improve learning results.  
- The usage of other network architectures such as
  - Trust Region Policy Optimization (TRPO)
  - and Truncated Natural Policy Gradient (TNPG)
  - Policy Optimization (PPO)
  - Distributed Distributional Deterministic Policy Gradients (D4PG)
- Utilize hyper parameter tuning
- Try different network architectures
  - converting to CNN, taking spatial aspects into account
  - different layer sizes and numbers of layers
  - try multiple activation layer types
  - add a head network to the input, splitting up the current state and the added prev_state-state input vector
- Implement an architecture utilizing Docker Containers for both hyper-parameter tuning as well as training/capturing multiple shared experiences.

