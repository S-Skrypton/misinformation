# Misinformation Spread Simulator

## 1. Node
For each node, it represents a user and it has several features:
1. followers list
2. following list
3. type (there are 3 types of users in this network: regular people, robot, celebrity)
4. repost probability: when this user posts a new post, the probability that this post will be reposted by his followers. regular: 5%; robot: 5%; celebrity: 10%

## 2. Graph
We randomly pick a node as a starting point (root of the tree). The resulting graph will be a tree structure.

## 3. Simulation Process
We first randomly generate 100 nodes, assign them as 3 types of people. Let's say each celebrity has at least 40% of the total network number of followers, each regular user has no more than 10% followers, each robot has no more than 10% followers. For the followings, robot and regular people cannot exceed 20% of the network. The following of a celebrity cannot exceed 5% of the network. For the whole population, 10% of the network is celebrity, 10% is robot, and the rest are regular people. The network is static when the misinformation spreads.

We randomly pick a node as a starting point, post an information and starting to generate the spreading tree. The simulation stopped if the whole network is infected or it can no longer be further reposted.

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

Their corresponding cost:
$0, e, e^2, e^3$

## 7. RL training input
experience tuple:
\[
(s, a, r, s')
\]
$s$ is the current node, $a$ is the current action, $r$ is the reward function, $s'$ is the next node on the spreading path.
\[
    r = cost(a) + cost(next \ node's \ type)
\]