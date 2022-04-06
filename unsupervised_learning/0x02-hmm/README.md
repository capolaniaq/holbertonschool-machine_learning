# 0x02. Hidden Markov Models

Author: Carlos Andres Polania (capolaniaq@correo.udistrital.edu.co)

## Learning Objectives

-   What is the Markov property?
-   What is a Markov chain?
-   What is a state?
-   What is a transition probability/matrix?
-   What is a stationary state?
-   What is a regular Markov chain?
-   How to determine if a transition matrix is regular
-   What is an absorbing state?
-   What is a transient state?
-   What is a recurrent state?
-   What is an absorbing Markov chain?
-   What is a Hidden Markov Model?
-   What is a hidden state?
-   What is an observation?
-   What is an emission probability/matrix?
-   What is a Trellis diagram?
-   What is the Forward algorithm and how do you implement it?
-   What is decoding?
-   What is the Viterbi algorithm and how do you implement it?
-   What is the Forward-Backward algorithm and how do you implement it?
-   What is the Baum-Welch algorithm and how do you implement it?

## Tasks

### 0. Markov Chain

Write the function `def markov_chain(P, s, t=1):` that determines the probability of a markov chain being in a particular state after a specified number of iterations:

### 1. Regular Chains

Write the function `def regular(P):` that determines the steady state probabilities of a regular markov chain:

### 2. Absorbing Chains

Write the function `def absorbing(P):` that determines if a markov chain is absorbing:

### 3. The Forward Algorithm
Write the function `def forward(Observation, Emission, Transition, Initial):` that performs the forward algorithm for a hidden markov model:
### 4. The Viretbi Algorithm
Write the function `def viterbi(Observation, Emission, Transition, Initial):` that calculates the most likely sequence of hidden states for a hidden markov model:
### 5. The Backward Algorithm
Write the function `def backward(Observation, Emission, Transition, Initial):` that performs the backward algorithm for a hidden markov model:

### 6. The Baum-Welch Algorithm

Write the function `def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):` that performs the Baum-Welch algorithm for a hidden markov model:
