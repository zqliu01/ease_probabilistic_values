"""
Basic usage example for easeshap package
"""
import numpy as np
from easeshap import runEstimator


def example_game_function(**game_args):
    """Example game function for demonstration"""
    # Define your game logic here
    pass


def main():
    # Configure parameters
    num_players = 10
    
    # Run exact computation
    estimator = runEstimator(
        estimator='exact_value',
        n_process=4,
        semivalue='shapley',
        semivalue_param=None,
        game_func=example_game_function,
        game_args={},
        num_player=num_players,
        nue_avg=1000,
        nue_per_proc=100,
        nue_track_avg=100,
    )
    
    values, trajectory = estimator.run()
    
    print("Computed semivalue:", values)
    print("Trajectory shape:", trajectory.shape)


if __name__ == "__main__":
    main()
