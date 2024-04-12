# Misinformation Spread Simulator

## 1. Node
For each node, it represents a user and it has several features:
1. followers list
2. following list
3. type (there are 3 types of users in this network: common people, robot, celebrity)
4. repost probability: when this user posts a new post, the probability that this post will be reposted by his followers. common: 5%; robot: 1%; celebrity: 15%

## 2. Graph
We randomly pick a node as a starting point (root of the tree). The resulting graph will be a tree structure.

## 3. Simulation Process
We first randomly generate 430 nodes, assign them as 3 types of people. Let's say each celebrity has at least 20% and at most 30% of the total network number of followers, each common user has no more than 10% ( or 100 if the network is very big) followers, each robot has no more than 10% ( or 100 if the network is very big) followers. For the followings, robot and common people cannot exceed 20% ( or 50 if the net work is very big) of the network. The following of a celebrity cannot exceed 5% (or 10 if the network is very big) of the network. For the whole population, 10% of the network is celebrity (100 if the network is very big), 10% is robot, and the rest are common people. The network is static when the misinformation spreads.

We randomly pick a node as a starting point, post an information and starting to generate the spreading tree. The simulation stopped if the whole network is infected or it can no longer be further reposted.

### 3.1 notice
We need to modify our parameters based on some human social networks study

## 4. Simulation Result
The result should be a tree structure, and a traverse function is needed to show all the spreading paths.

## 5. Python Package
We use the NetworkX package.

## 6. Action
There are 4 types of action we can take to intervene with the misinformation:
0: no action
1: reduce 25% of the repost probability
2: reduce 50% of the repost probability
3: reduce 100% of the repost probability
note: action should be attribute of the poster

Their corresponding cost: (exponentially increasing)
$0, 10, 10^2, 10^3$

## 7. RL training input
experience tuple:
\[
(s, a, r, s')
\]
$s$ is the current node, $a$ is the current action, $r$ is the reward function, $s'$ is the next node on the spreading path.
\[
    r = -(cost(a) + cost(next \ node's \ type))
\]

## 8. Baseline
take one action all the time and compare with our result