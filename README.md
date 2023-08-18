# AI_algorithm
Ideas and techniques underlying the design of intelligent computer systems. Topics include search, game playing, knowledge representation, inference, planning, reasoning under uncertainty, machine learning, robotics, perception, and language understanding.

** This repo only includes sample code and algorithms written by Zilong. The project idea are from the UCB CS188 class.

---

## Project 1 Search:
The Pacman agent will find paths through his maze world, both to reach a particular location and to collect food efficiently. I built general search algorithms and applied them to Pacman scenarios.

## Project 2 Logic and Classical Planning:
I used simple Python functions that generate logical sentences describing Pacman physics, aka pacphysics. Then I used a SAT solver, pycosat, to solve the logical inference tasks associated with planning (generating action sequences to reach goal locations and eat all the dots), localization (finding oneself in a map, given a local sensor model), mapping (building the map from scratch), and SLAM (simultaneous localization and mapping).

## Project 3 Multi-Agent Search:
I designed agents for the classic version of Pacman, including ghosts. I implemented both minimax and expectimax search and evaluation function designs along the way.

## Project 4 Reinforcement Learning:
I implemented value iteration and Q-learning and test my agents first on Gridworld, then applied them to a simulated robot controller (Crawler) and Pacman.

## Project 5 Ghostbusters:
I designed Pacman agents that use sensors to locate and eat invisible ghosts. I advanced from locating single, stationary ghosts to hunting packs of multiple moving ghosts with ruthless efficiency.

## Project 6 Machine Learning:
I built a neural network to classify digits in this project.

---

Language: Python

Compiler: Visual Studio Code

Author: Zilong Guo

Date: February - May 2023
