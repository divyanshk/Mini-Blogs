---
layout: default
---

[Deconvolution Layer](#deconv)  
[Batch Normalization](#batchnorm)  
[SqueezeNet](#squeezenet)  
[Q-learning v SARSA](#qlearningsarsa)  
[Policy Iteration v Value Iteration](#policyvalue)   
[Q Learning](#qlearning)  
[Policy Gradients](#policygrad)  
[Actor Critic methods](#actorcritic)  
[Trust Region Methods](#trpo)  
[Monte Carlo Tree Search](#mcts)  
[Inverse Reinforcement Learning](#irl)  

---

## <a name="deconv"></a>Deconvolution Layer

* torch.nn.ConvTranspose2d in PyTorch
* ambiguous name, no deconvolutions
* a deconvolution layer maps from a lower to higher dimension, a sort of upsampling
* the transpose of a non-padded convolution is equivalent to convolving a zero-padded input
* zeroes are inserted between inputs which cause the kernel to move slower, hence also called fractionally strided convolution
* deconv layers allow the model to use every point in the small image to “paint” a square in the larger one
* deconv layers have uneven overlap in the output, conv layers have overlap in the input
* leads to the problem of checkerboard artifacts
* resize-convolution instead transposed-convolution to avoid checkerboard artifacts

References
* [Convolution Arithmatic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html){:target="_blank"}
* [Distil Blog Post](https://distill.pub/2016/deconv-checkerboard/)
* [Original Paper](http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf)

---

## <a name="batchnorm"></a>Batch Normalization

* torch.nn.BatchNorm2d in PyTorch
* normalizses the data in each batch to have zero mean and unit covariance
* provides some consistency between layers by reducing internal covariate shift
* allows a higher learning rate to be used, reduces the learning time
* after normalizing the input, it is squased through a linear function with parameters gamma and beta
* output of batchnorm = gamma * normalized_input + beta
* having gamma and beta allows the network to choose how much 'normalization' it wants for every feature; shift and scale

References
* [Andrej Karapathy's lecture](https://www.youtube.com/watch?v=gYpoJMlgyXA&feature=youtu.be&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&t=3078)
* [Original Paper](https://arxiv.org/abs/1502.03167)
* [Read this later](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

---

## <a name="squeezenet"></a>SqueezeNet

* A conv net aimed at drastically reducing the number of parameter without much loss to accuracy
* Uses 1x1 filters instead of 3x3
* 1x1 filters work by capturing information on the channels of the pixels as compared to the neighborhood
* Uses a 'fire' module, composed of 'squeeze' and 'expand' layers
* The number of filters in the squeeze layer is restricted, thereby reducing the input channels to the expand layer
* Downsampling late into the network to have larger activation maps
* No fc layers

References
* [Original paper](https://arxiv.org/pdf/1602.07360.pdf)
* [KDNuggest summary](https://www.kdnuggets.com/2016/09/deep-learning-reading-group-squeezenet.html)

---

## <a name='qlearningsarsa'></a>Q-learning v SARSA

* SARSA stands for state-action-reward-state-action
* SARSA is on-policy; that is sticks to the policy it is learning. Q-learning is off-policy
* SARSA improves the estimate of Q by using the transitions from the policy dervied from Q 
* Q-learning updates the Q estimate using the observed reward and the maximum reward possible $$ max_a{a\prime} Q(s\prime, a\prime) $$ for the next state


References
* [Pseudo Codes](http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html)
* [StackOverFlow](https://stackoverflow.com/questions/32846262/q-learning-vs-sarsa-with-greedy-select)

---

## <a name='policyvalue'></a>Policy Iteration v Value Iteration

* PI: trying to converge the policy to optimal; VI: trying to converge the value function to optimal
* PI: policy evaluation (calculating value function using $$ v(s) \gets \sum_{s\prime} p(s\prime \mid s, \pi (s)) [r(s, \pi (s), s\prime) + \gamma v(s\prime)] $$) ) + policy improvement; repeat until policy is stable
* VI: policy evaluation (calculating value function using $$ v(s) \gets max_a \sum_{s\prime} p(s\prime \mid s,a) [r(s,a,s\prime) + \gamma v(s\prime)] $$); single policy update

References
* [StackOverFlow](https://stackoverflow.com/questions/37370015/what-is-the-difference-between-value-iteration-and-policy-iteration)

---

## <a name='qlearning'></a>Q Learning

* Model free learning: the agent has no idea about the state transition and reward functions; it learns everything from experience by interacting with the environment
* Q-Learning is based on Time-Difference Learning
* $$ Q(s_t, a_t) = Q(s_t, a_t) + \alpha[r(s,a) + \gamma * max_a Q(s_{t+1}, a) - Q(s_t, a_t)] $$
* See notes on Q-Learning v SARSA
* $$ \epsilon $$-greedy approach: choose a random action with probability $$ \epsilon $$, or action according to the current estimate of Q-values otherwise; this approach controls the exploration vs exploitation

References
* Sutton Book 1st Ed Page 148
* [Medium post](https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa)

---
## <a name='policygrad'></a>Policy Gradients

* Run a policy for a while; see what actions led to higher rewards; increase their probability
* Take the gradient of log probability of trajectory, then weight it by the final reward
* Increase the probability of actions that lead to higher reward
* With $$ J(\theta) $$ as the policy objective function	
$$ \nabla_{\theta} J(\theta) = \sum_{t \geq 0} r(\tau) \;\nabla_{\theta} \;log\; \pi_{\theta} (a_t \mid s_t) $$
* This suffers from high variance and is a simplistic view; credit assignment problem is hard 
* Baseline: whether a reward is better or worse than what you expect to get
* A simple baseline: constant moving average of rewards experienced so far from all trajectories; Vanilla REINFORCE
* Reducing variance further using better baselines -> Actor critic algorithm

References
* [CS231n RL Lecture](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf)
* [David Silver slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)

---

## <a name='actorcritic'></a>Actor Critic Methods

* Works well when there is an infinite input and output space
* Requires much less training time than policy gradient methods
* Actor => takes in the environment states and determines the best action to take
* Critic => takes in the environment and the action from the actor and returns a score that represents how good the action is for the state
* Both the actor (policy) and critic (Q function) are different neural networks

References
* [CS231n RL Lecture](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf)
* [CS294 DeepRL](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_5_actor_critic_pdf.pdf)

---

## <a name='trpo'></a>Trust Region Methods

* A kind of local policy search algorithm
* 'local' because every new policy is somewhat closer to the earlier policy
* TRPO uses policy gradients but has a constraint on how the polices are updated
* Each new policy has to be close to the older one in terms of the KL-divergence
* Since polices are nothing but probability distributions over the actions, KL divergence is a natural way to measure the distance
* Constraint Policy Optimization (CPO) is another trust region method using contraints on the cost function to keep an agent's action under a limit while maintaining optimal performance

References
* [TRPO paper](https://arxiv.org/abs/1502.05477)
* [OpenAI Blog on CPO](http://bair.berkeley.edu/blog/2017/07/06/cpo/)

---

## <a name='mcts'></a>Monte Carlo Tree Search

* MCTS is based on two idea:
    * a true value of an action may be evaluated using random simulation
    * these values maybe used to efficiently adjust the policy towards a best-first strategy
* THe algorithm builds a search tree till a computational budget - time or memory is exhausted
* The algorthim has four parts which are applied per iteration
    * Selection: descending down the root node till an expandable non-terminal node 
    * Expansion: adding child nodes towards the tree
    * Simulation: simulate the default policy from the new node(s) to produce an output
    * Backpropagation: the simulation result is 'backed up' through the selected nodes
* Selection + Expansion => Tree policy; Simluation => Default policy
* The backpropagatin step informs future tree policy decision

References
* [MCTS Survey paper](https://gnunet.org/sites/default/files/Browne%20et%20al%20-%20A%20survey%20of%20MCTS%20methods.pdf)
* Sutton 2nd Edition 8.11 Page 153

---

## <a name='irl'></a>Inverse Reinforcement Learning

* Learning the reward fucntion by observing expert behaviour
* Imitation learning or behaviour cloning tries to copy the teacher's actions
* Learning the reward function can make the system robust to changes in the environment's transition mechanics
* Learning the reward function is also transferable from one type of agent to another, as it encodes all that is needed to excel in the envirnment
* Think of IRL as a way to learn an abstraction or latent representation of the target
* Another big motivation for IRL is that it is extremely difficult to manually specifiy a reward function to an agent, like in a self driving car
* Instead of simply copying the expert behavior, we can then try to learn the underlying reward function which the expert is trying to optimize

References
* [Blog post](https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html)

---
