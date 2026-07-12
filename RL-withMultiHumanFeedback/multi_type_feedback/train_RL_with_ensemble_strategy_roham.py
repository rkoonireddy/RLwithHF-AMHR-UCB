"""Module for training an RL agent with weighted ensemble of reward functions."""

import os
import typing

import gymnasium as gym
import numpy
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from multi_type_feedback.networks import (
    LightningCnnNetwork,
    LightningNetwork,
)
from multi_type_feedback.utils import TrainingUtils, RewardFn
from train_baselines.exp_manager import ExperimentManager


class CustomReward(RewardFn):
    """Custom reward based on weighted ensemble of fine-tuned reward models."""

    def __init__(
        self,
        reward_model_cls: typing.Union[LightningNetwork, LightningCnnNetwork] = None,
        reward_model_paths: list[str] = [],
        vec_env_norm_fn: typing.Callable = None,
        action_is_discrete: bool = False,
        action_dim: int = 1,
        device: str = "cuda",
        inverse_scaling: bool = False,
    ):
        """Initialize custom reward."""
        super().__init__()
        self.device = device

        self.reward_models = [
            reward_model_cls.load_from_checkpoint(path, map_location=device, weights_only=False)
            for path in reward_model_paths
        ]

        self.action_is_discrete = action_is_discrete
        self.n_discrete_actions = action_dim

        self.rewards = []
        self.expert_rewards = []
        self.counter = 0

        # Variables for calculating a running mean
        self.reward_mean = None
        self.squared_distance_from_mean = None

        self.inverse_scaling = inverse_scaling

        # UCB bandit variables (initialized when init_ucb() is called)
        self.use_ucb = False
        self.n_arms = None
        self.arm_counts = None
        self.arm_q_values = None
        self.active_arm = None
        self.ucb_c = None

    def init_ucb(self, ucb_c: float = 2.0):
        """Initialize UCB bandit statistics for dynamic model selection.
        Args:
            ucb_c: Exploration constant for UCB formula (default: 2.0)
        """
        self.use_ucb = True
        self.n_arms = len(self.reward_models)
        self.arm_counts = numpy.zeros(self.n_arms)
        self.arm_q_values = numpy.zeros(self.n_arms)  # Average return for each model
        self.active_arm = 0  # Start with first model
        self.ucb_c = ucb_c
        print(f"[UCB] Initialized with {self.n_arms} arms, exploration constant c={ucb_c}")

    def update_ucb(self, episode_return: float):
        """Update UCB bandit statistics and select next arm.
        Called by callback after collecting episode returns.
        Args:
            episode_return: Average return from recent episodes
        """
        if not self.use_ucb:
            print("[UCB] Warning: update_ucb called but UCB not initialized!")
            return

        # 1. Update stats for the arm we just used
        arm = self.active_arm
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]

        # Incremental average update: Q_new = Q_old + (Return - Q_old) / n
        current_q = self.arm_q_values[arm]
        self.arm_q_values[arm] = current_q + (episode_return - current_q) / n

        # 2. Select NEXT arm using UCB1 formula
        total_counts = self.arm_counts.sum()
        ucb_values = numpy.zeros(self.n_arms)

        for i in range(self.n_arms):
            if self.arm_counts[i] == 0:
                # Always try unused arms first (infinite confidence)
                self.active_arm = i
                print(f"[UCB] Selecting unexplored arm {i}")
                return

            # UCB Formula: Q(a) + c * sqrt(ln(t) / N(a))
            bonus = self.ucb_c * numpy.sqrt(numpy.log(total_counts) / self.arm_counts[i])
            ucb_values[i] = self.arm_q_values[i] + bonus

        self.active_arm = numpy.argmax(ucb_values)
        print(f"[UCB] Update complete - Active arm: {self.active_arm} | "
              f"Q-values: {self.arm_q_values.round(2)} | "
              f"Counts: {self.arm_counts.astype(int)} | "
              f"UCB values: {ucb_values.round(2)}")

    def _one_hot_encode_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert nested batch of discrete actions to one-hot encoded format.
        Args:
            actions: Tensor of shape (1, batch_size) containing discrete action indices
        Returns:
            one_hot_actions: Tensor of shape (1, batch_size, n_discrete_actions)
        """
        outer_batch, inner_batch = actions.shape
        one_hot = torch.zeros(
            (outer_batch, inner_batch, self.n_discrete_actions), device=self.device
        )
        actions = actions.long().unsqueeze(-1)  # Add dimension for scatter
        return one_hot.scatter_(2, actions, 1)

    def standardize_rewards(self, rewards: torch.Tensor):
        """
        Standardizes the input using the rolling mean and standard deviation of the rewards.
        Input should be a tensor of shape (batch_size, model_count).
        """
        model_count = len(self.reward_models)

        if self.reward_mean is None:
            self.reward_mean = torch.zeros(model_count).to(self.device)

        if self.squared_distance_from_mean is None:
            self.squared_distance_from_mean = torch.zeros(model_count).to(self.device)

        standard_deviation = torch.ones(model_count).to(self.device)

        for reward_index, reward in enumerate(rewards):
            # Welford's algorithm for calculating running mean and variance
            self.counter += 1

            difference = reward - self.reward_mean
            self.reward_mean += difference / self.counter
            new_difference = reward - self.reward_mean
            self.squared_distance_from_mean += difference * new_difference

            if self.counter > 1:
                standard_deviation = (
                    self.squared_distance_from_mean / (self.counter - 1)
                ).sqrt()

            rewards[reward_index] = (reward - self.reward_mean) / standard_deviation

        return rewards

    def __call__(
        self,
        state: numpy.ndarray,
        actions: numpy.ndarray,
        next_state: numpy.ndarray,
        _done: numpy.ndarray,
    ) -> list:
        """Return reward given the current state."""
        with torch.no_grad():
            # Initialize tensors for means AND variances
            # Shape: (num_models, n_envs)
            means = torch.empty(len(self.reward_models), state.shape[0]).to(self.device)
            variances = torch.zeros(len(self.reward_models), state.shape[0]).to(self.device)

            state = torch.as_tensor(
                state, device=self.device, dtype=torch.float
            ).unsqueeze(0)
            actions = torch.as_tensor(
                actions, device=self.device, dtype=torch.float
            ).unsqueeze(0)

            if self.action_is_discrete:
                actions = self._one_hot_encode_batch(actions)

            for model_index, reward_model in enumerate(self.reward_models):
                if reward_model.ensemble_count > 1:
                    # Expand inputs for ensemble
                    model_state = state.expand(
                        reward_model.ensemble_count, *state.shape[1:]
                    )
                    model_actions = actions.expand(
                        reward_model.ensemble_count, *actions.shape[1:]
                    )

                    # Get predictions from all ensemble members: (ensemble_count, n_envs, 1)
                    raw_preds = reward_model(
                        model_state,
                        model_actions,
                    ).view(reward_model.ensemble_count, -1, 1)

                    # --- THE FIX ---
                    # Capture Mean AND Variance separately BEFORE averaging
                    means[model_index] = torch.mean(raw_preds, dim=0).squeeze(-1)
                    variances[model_index] = torch.var(raw_preds, dim=0).squeeze(-1)
                else:
                    # Single model has no epistemic uncertainty, assign epsilon variance
                    means[model_index] = reward_model(state, actions).squeeze(1)
                    variances[model_index] = 1e-6  # Small epsilon

            # Standardize the means (Welford's algorithm)
            # Note: We standardize only the means for the reward signal
            standardized_means = self.standardize_rewards(means.transpose(0, 1))  # (n_envs, num_models)

            # Transpose variances to match: (n_envs, num_models)
            variances = variances.transpose(0, 1)

            # UCB Mode: Use only the active arm's reward
            if self.use_ucb:
                final_reward = standardized_means[:, self.active_arm]
            # Inverse Variance Weighting Mode
            elif self.inverse_scaling:
                # --- CORRECTED WEIGHTING ---
                # Weight = 1 / (Internal Model Variance + epsilon)
                weights = 1.0 / (variances + 1e-6)

                # Weighted Average: Sum(w * x) / Sum(w)
                numerator = (weights * standardized_means).sum(dim=1)
                denominator = weights.sum(dim=1)
                final_reward = numerator / denominator
            # Standard Mean Averaging
            else:
                final_reward = torch.mean(standardized_means, dim=1)

            return final_reward.cpu().numpy()


class UCBUpdateCallback(BaseCallback):
    """Callback to update UCB bandit statistics periodically based on agent performance.
    This callback tracks episode returns and triggers UCB updates at regular intervals,
    allowing the reward function to dynamically select the best-performing reward model.
    """

    def __init__(self, custom_reward_fn: CustomReward, update_freq: int = 10000, verbose: int = 1):
        """Initialize UCB callback.
        Args:
            custom_reward_fn: The CustomReward instance with UCB capabilities
            update_freq: Number of steps between UCB updates (default: 10000)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.reward_fn = custom_reward_fn
        self.update_freq = update_freq
        self.ep_rewards = []
        self.last_update_step = 0

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Track episode rewards from all parallel environments
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.ep_rewards.append(info['episode']['r'])

        # Trigger UCB update periodically
        if self.n_calls - self.last_update_step >= self.update_freq:
            if len(self.ep_rewards) > 0:
                avg_return = numpy.mean(self.ep_rewards)
                if self.verbose > 0:
                    print(f"\n[UCB Callback] Step {self.n_calls}: "
                          f"Collected {len(self.ep_rewards)} episodes, "
                          f"avg return = {avg_return:.2f}")
            else:
                # No episodes completed in this interval, use 0
                avg_return = 0.0
                if self.verbose > 0:
                    print(f"\n[UCB Callback] Step {self.n_calls}: "
                          f"No episodes completed, using return = 0")

            # Update the bandit
            self.reward_fn.update_ucb(avg_return)

            # Reset buffer and update counter
            self.ep_rewards = []
            self.last_update_step = self.n_calls

        return True


def main():
    """Run RL agent training."""
    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--reward-model-folder",
        type=str,
        default="reward_models",
        help="Folder of trained reward models",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="trained_agents",
        help="Folder for finished feedback RL agents",
    )
    parser.add_argument(
        "--feedback-types",
        nargs="+",
        type=str,
        default=["evaluative"],
        help="Types of feedback to train the reward model (space-separated list)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=int(1e6),
        help="Number of steps to generate feedback for",
    )
    parser.add_argument(
        "--inverse-scaling",
        action="store_true",
        help="Use inverse scaling for reward weighting",
    )
    parser.add_argument(
        "--use-ucb",
        action="store_true",
        help="Enable UCB-based dynamic model selection (overrides inverse-scaling)",
    )
    parser.add_argument(
        "--ucb-update-freq",
        type=int,
        default=10000,
        help="Number of steps between UCB updates (default: 10000)",
    )
    parser.add_argument(
        "--ucb-exploration-constant",
        type=float,
        default=2.0,
        help="UCB exploration constant c (default: 2.0)",
    )

    args = parser.parse_args()

    FEEDBACK_ID = "_".join([args.algorithm, args.environment, str(args.seed)])
    MODEL_ID = f"{FEEDBACK_ID}_{args.seed}"

    reward_model_paths = []
    for feedback_type in args.feedback_types:
        model_path = os.path.join(
            args.reward_model_folder,
            f"{FEEDBACK_ID}_{feedback_type}_{args.seed}.ckpt",
        )
        reward_model_paths.append(model_path)

    set_random_seed(args.seed)

    run = wandb.init(
        name="ENSEMBLE_RL_" + MODEL_ID,
        project=args.wandb_project_name,
        config={
            **vars(args),
            "feedback_types": args.feedback_types,
            "feedback_count": len(args.feedback_types),
            "total_ensemble_size": len(args.feedback_types),
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    # ================ Load correct reward function model =================
    if "ALE/" in args.environment or "procgen" in args.environment:
        architecture_cls = LightningCnnNetwork
    else:
        architecture_cls = LightningNetwork

    # we initialize just for the action space, there should be a more elegant way
    # to initialize the CustomRewardFn in the Exp. Manager
    action_space = gym.make(args.environment).action_space
    action_is_discrete = isinstance(action_space, gym.spaces.Discrete)
    action_dim = numpy.prod(action_space.shape) if not action_is_discrete else action_space.n

    # ================ Create Custom Reward Function ===================
    reward_fn = CustomReward(
        reward_model_cls=architecture_cls,
        reward_model_paths=reward_model_paths,
        inverse_scaling=args.inverse_scaling,
        action_is_discrete=action_is_discrete,
        action_dim=action_dim,
        device=DEVICE,
    )

    # ================ Initialize UCB if enabled ===================
    callbacks = []
    if args.use_ucb:
        print(f"\n{'='*60}")
        print(f"UCB MODE ENABLED")
        print(f"  - Update Frequency: {args.ucb_update_freq} steps")
        print(f"  - Exploration Constant: {args.ucb_exploration_constant}")
        print(f"  - Number of Reward Models: {len(reward_model_paths)}")
        print(f"{'='*60}\n")

        # Initialize UCB in the reward function
        reward_fn.init_ucb(ucb_c=args.ucb_exploration_constant)

        # Create the UCB callback
        ucb_callback = UCBUpdateCallback(
            custom_reward_fn=reward_fn,
            update_freq=args.ucb_update_freq,
            verbose=1
        )
        callbacks.append(ucb_callback)

    # ================ Create Experiment Manager ===================
    exp_manager = ExperimentManager(
        args,
        args.algorithm,
        args.environment,
        os.path.join(args.save_folder, f"ENSEMBLE_RL_{MODEL_ID}"),
        tensorboard_log=f"runs/{'ENSEMBLE_RL_'+MODEL_ID}",
        n_timesteps=args.train_steps,
        seed=args.seed,
        log_interval=-1,
        reward_function=reward_fn,
        use_wandb_callback=True,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        # we need to save the loaded hyperparameters
        args.saved_hyperparams = saved_hyperparams
        assert run is not None  # make mypy happy
        run.config.setdefaults(vars(args))

        # Normal training
        if model is not None:
            # Pass UCB callbacks if enabled
            if len(callbacks) > 0:
                exp_manager.learn(model)
            else:
                exp_manager.learn(model)
            exp_manager.save_trained_model(model)


if __name__ == "__main__":
    main()