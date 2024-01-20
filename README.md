# Keep_Talking_and_Nobody_Explodes
Source codes of the course project *Keep Talking and Nobody Explodes: A New Design of Dynamic Environment for Emergent Languages* by Core123 in Cognitive and Reasoning (2023 Fall).

The brief introduction concerning the structure of files:

- **src/supervised-wire**：source code of the supervised method on *Wire Riddles* environment. For training, run `gumbel_ae.py` with arguments specified in `config.py`. This will automatically record the metrics information every 100 episode, and save the modules that succeed in predicting all action profiles given the batch of initial state profiles. The logging directory and saving directory should be carefully specified (see `config.py`). 
- **src/rl-wire**：source code of the RL method on *Wire Riddles* environment.
- **src/rl-maze**：source code of the RL method on *Maze Mastermind* environment.
- **rendering**：source code of our visualization of *Wire Riddles* environment.

