# Pacman Project - CS188: Introduction to Artificial Intelligence

## Overview
This project is part of the CS188: Introduction to Artificial Intelligence course at UC Berkeley. It focuses on implementing various AI techniques to enable Pacman to navigate mazes, collect food, and avoid ghosts. The project is divided into multiple assignments, each exploring different AI concepts such as search algorithms, reinforcement learning, and adversarial planning.

## Project Structure
The project consists of multiple parts, each addressing a specific aspect of AI:

1. **Search (Project 1)**: Implement search algorithms (DFS, BFS, UCS, A*) to find optimal paths in the Pacman world.
2. **Multi-Agent (Project 2)**: Develop adversarial agents using minimax, alpha-beta pruning, and expectimax.
3. **Reinforcement Learning (Project 3)**: Implement Q-learning and approximate Q-learning to train Pacman through experience.
4. **Bayes Nets (Project 4)**: Apply probabilistic inference to track ghosts using hidden Markov models.
5. **Machine Learning (Project 5)**: Use perceptron and neural networks to classify digits.

## Requirements
- Python 3.x
- NumPy
- Matplotlib (for visualization)
- Additional dependencies as required by specific projects (e.g., SciPy for machine learning)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pacman-project.git
   cd pacman-project
   ```
2. Ensure Python dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project
Each part of the project includes specific scripts to run different AI algorithms:

- **Search Algorithms**:
  ```bash
  python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
  ```
  (Replace `bfs` with `dfs`, `ucs`, or `astar` as needed.)

- **Multi-Agent Algorithms**:
  ```bash
  python pacman.py -p MinimaxAgent -l trappedClassic -a depth=2
  ```

- **Reinforcement Learning**:
  ```bash
  python pacman.py -p QLearningAgent -x 2000 -n 2010 -l smallGrid
  ```

Refer to each project’s `README` file for more detailed usage instructions.

## Evaluation
Each project comes with an autograder to test implementations:
```bash
python autograder.py
```

## Contributors
This project is based on the UC Berkeley CS188 Pacman AI framework.

## License
This project is for educational purposes only and follows the policies outlined by UC Berkeley's CS188 course.

