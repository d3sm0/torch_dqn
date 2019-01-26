## (Deep) Reinforcement Learning

From an italian plumber to a super intelligent agent

Note:
- how many rl people that read about it, that tried, that want to make research on it?
- how many never heard or tried deep learning before I will use a little of gergo around the end of the talk. 

---
## Outline
- What is (Deep) Reinforcement Learning?
- Where does it come from?
- How does it work?
- What can I do with it?
- Time to code!


### Ready?


![mario](img/rl/mario-bros.jpg)

---
## Hype first


### Atari 2600
![mario](img/rl/space_invaders.gif)

[Mnih et al, 2015](https://www.nature.com/articles/nature14236)


### AlphaZero
![alpha_go](img/rl/alpha_go.jpg)

[Silver et al, 2018](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)


![1up](img/rl/1up.jpg)


### Libratus
![lbratus](img/rl/libratus.jpg)

[Brown, Sandholdm, 2018](http://science.sciencemag.org/content/359/6374/418)


### Autonomous Driving
![car](img/rl/torcs.jpg)

[Wang et al, 2018](https://arxiv.org/pdf/1811.11329.pdf)


### Financial Trading
![stock](img/rl/stock.png)

[Jiang et al, 2017](https://arxiv.org/pdf/1706.10059.pdf)

---
## Intuition


![dog](img/rl/dog_teaching.jpeg)

Note:
Have you ever had a dog? Anyhow, a basic way to teach him to get the paw (pata) is to ask for it and give him a little reward everytime it perform the correct action. Eventually your dog will give you back the paw anytime you ask for it. This is the principle of conditional stimulus


![conditioning](img/rl/conditioning.png)


### Notice
We have three key components: 
1. conditioning stimilus
2. prediction of what could happen next
3. optimal response


### Questions?
![questions](img/rl/questions.png)

---
### 8 Bit of Math


A Markov Decision Process (MDP) is a tuple `$(S, A, R, P, \gamma)$`


- S: State-space, discrete or continous
- A: Action-space, discrete or continous
- R: Function from the real to the real
- P: Transition probability that is `$P(s_{t+1} \rvert s_{t}, a_t)$`

Note:
- Markov means markov property, that is the state that we observe is enough to make all the decision for the best action
- The reward function can be anything and it must provide the signal for the goodness of the performance. It must be bounded.


![mdp](img/rl/mdp.png)


Let a policy be  `$\pi(a \rvert s)$`

Note:
- deterministic 
- stochastic


The objective of algorithm is to maximize the cumulative reward, that is:

`$$ E \sum_{t=0}^T r(s_t, a_t) $$`

Note:
- why there is an expectation? because we have stocahstic in P for sure
- but not only


In particular we care about the discounted reward over an infinite horizon:

`$$ E \sum_{t=0}^\infty \gamma^t r(s_t, a_t) $$`

Note:
- Question who can tell me what's going on here?
- Why we have a gamma?


Which by we can write as:

`$$ E r(s_0, a_0) + \sum_{t=1}^\infty \gamma^t r(s_t, a_t) $$`

Note:
- We can write this as a recursive formula as follows. Where the tail is actually the future discounted reward. Which is exactly what we care about. If you think about it. We can maximize the reward if and only if we collect the maximum reward in the future. 
- Remember that the only way to interact with an mdp is trough a policy. A policy is a way to choose action, therefore a more appropriate tool that can tell us how well is the agent doing 
- is a called a value function. A value function is of this form


![jump_mario](img/rl/jump_mario.jpg)


Which we can call as:

`$$ V^\pi(s) = E \sum_{l=t}^\infty \gamma^l r(s_t, \pi(a_t)) \rvert s $$`


Or by including all possibile actions, we can write:

`$$ Q^\pi(s',a) = r(s,a) + \gamma V^\pi(s') $$`

Note:
- What is this function telling me?
- Note the expectation now is not only on P but also on pi
- This last equation is telling me that i maximize my reward if i act greedly wrt to the futre
- This gives us an intuition about how we can solve an mdp


The optimal state-action value function is defined as:

`$$ Q^\ast(s,a) = max_\pi Q^\pi(s,a)$$` 

Note:
- we are looking for the policy that achieve the maximum future reward for each (s,a)


Which we know how to reach by looking one step into the future:
`$$ r(s,a) + \gamma max_{a \in A} Q(s', a) $$`


### Questions?
![questions](img/rl/questions.png)

---
## How to solve it?

### e.g. how to save the princess?


1. Evaluate the current policy
2. Find the action that maximize the future rewards
3. Update the future values given the best action


```{python}
def q(s):
  """
  Given a state, returns the future values for each possible action
  """
  # add mushrooms + neural nets
  return action_value
```

Note:
- how many values do i have


```{python}
best_action = np.argmax(action_value, axis=-1)

best_value = np.max(action_value, axis=-1)
```


```{python}
def loss(s, a, r, s1, gamma=.99):
  
  be = r + gamma * np.max(q(s1), axis=-1) - (a * q(s)).sum(axis=-1) # assume a is one-hot
  be = .5 * np.mean(be**2)
  return be
```


```{python}
s = env.reset()
done = False
beta = 0.1 # learning rate
while not done:
  action_value = q(s)
  best_action = np.argmax(action_value, axis=-1)
  s1, r, d, = env.step(best_action)
  error = loss(s,a,r,s1)
  optimize(error)
  s = s1
```


### Questions?
![questions](img/rl/questions.png)

---
## Time to go deeper...


```{python}
  class QNetwork(torch.nn.Module):
    def __init__(obs_shape, act_shape):
      self.fc = torch.nn.Linear(obs_shape, act_shape)
    def forward(s):
      return self.fc(s)
```


```{python}
  def policy(s, q, eps):
    best_action = np.argmax(q, axis=-1)
    if eps > 0.1:
      best_action = np.random.randint(0, q.shape[1])
    return best_action
```


```{python}
s = env.reset()
done = False
memory = Memory(max_size=int(1e5))
gamma = .99
while not done:
  best_action = policy(s,q,eps=0.1)
  s1, r, d, = env.step(best_action)
  memory.append((s,a,r,s1,d))
  s = s1
batch = memory.sample()
s,a,r,s1,d = batch

loss = .5 * ( (r + gamma * q(s1) - q(s) )**2).mean()
opt.zero_grad()
loss.backward()
opt.step()

```


Some hacks that made it work:
1. Target network
2. Experience Replay
3. Eps-greedy policy
4. Reward scaling


### Questions?
![questions](img/rl/questions.png)

---
## Secret mushrooms
- Start from the gridworld 
- Log everything
- Trust the math

---
## Bonus

### Theory
- Mongetes Màgiques [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)
- Kaioken [UCL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- Super sayan [Bertsekas](https://www.amazon.com/gp/product/1886529086/ref=dbs_a_def_rwt_bibl_vppi_i1)


### Code

### To learn...

- [spinup](https://github.com/openai/spinningup)
- [gym](https://github.com/openai/gym)
- [gridworld](https://github.com/maximecb/gym-minigrid)


### To cry...
- [baselines](https://github.com/openai/baselines)
- [starcraft](https://github.com/deepmind/pysc2)

---
### Contacts

simone.totaro@gmail.com

d3sm0.github.io

---
![mario_flag](img/rl/mario_flag.jpg)
