
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from ...discrete_control_reward_experiment.environment import make_env, SaveBestWithStats, CustomLoggingCallback

from torchinfo import summary
import torch
from torchviz import make_dot

from torch.utils.tensorboard import SummaryWriter

shape_vertices = {
    "Circle":          [],
    "Square":          [(-30, -30), (30, -30), (30, 30), (-30, 30)],
    "RA triangle":     [(-30, -30), (30, -30), (30, 30)],
    "Equi Triangle":   [(-30, -30), (30, -30), (0, 30)]}


class ParameterLogger(BaseCallback):
    def __init__(self, log_dir: str, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            # Iterate over all policy parameters
            for name, param in self.model.policy.named_parameters():
                self.writer.add_histogram(f"policy/{name}", param.detach().cpu().numpy(), self.num_timesteps)
        return True

def train_from_scratch(vertex, design_vector):

    N_ENVS = 1  # Number of parallel environments

    # Define the policy network architecture
    policy_kwargs = {'net_arch': [256, 256], "log_std_init": 2}

    # Training envs (headless, parallel)
    train_env = DummyVecEnv([make_env(vertex, i, design_vector) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Env for saving best model
    eval_env = DummyVecEnv([make_env(vertex, 0, design_vector, render=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # Copy the observation normalization stats from the training env to the eval env
    eval_env.obs_rms = train_env.obs_rms

    best_ckpt = SaveBestWithStats(
        eval_env,
        vecnormalize=train_env,
        number=1,
        best_model_save_path=f"pymunk_experiments/gripper2/discrete_control/RL_transfer_experiments/1/models/ppo_pymunk_gripper_best",
        n_eval_episodes=5,
        eval_freq=20_000,
        deterministic=True,
        render=False,
        verbose=0,

    )

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=500000, render=True, verbose=0,
                                 deterministic=True)

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,  # 256 × 8 = 2048 steps / update
        batch_size=256,  # must divide N_ENVS × N_STEPS
        verbose=0,
        tensorboard_log="pymunk_experiments/gripper2/discrete_control/RL_transfer_experiments/ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        learning_rate=1e-3,
    )

    custom_logger = CustomLoggingCallback()

    model.learn(total_timesteps=3000000, callback=[eval_callback, best_ckpt, custom_logger])

    print("Training complete and model saved")

    train_env.close()
    eval_env.close()

def train_from_similar(vertex, design_vector):

    N_ENVS = 1  # Number of parallel environments

    # Define the policy network architecture
    policy_kwargs = {'net_arch': [256, 256], "log_std_init": 2}

    # Training envs (headless, parallel)
    train_env = DummyVecEnv([make_env(vertex, i, design_vector) for i in range(N_ENVS)])
    train_env = VecNormalize.load(f"pymunk_experiments/gripper2/discrete_control/RL_transfer_experiments/1/normalise_stats/vecnormalize_stats_best.pkl", train_env)
    train_env.training = True

    model = PPO.load(f"pymunk_experiments/gripper2/discrete_control/RL_transfer_experiments/1/models/ppo_pymunk_gripper_best/best_model", env=train_env, tensorboard_log="pymunk_experiments/gripper2/discrete_control/RL_transfer_experiments/ppo_gripper_tensorboard/")

    # for pg in model.policy.optimizer.param_groups:
    #     pg['lr'] = 0.00001

    # Env for saving best model
    eval_env = DummyVecEnv([make_env(vertex, 0, design_vector, render=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # Copy the observation normalization stats from the training env to the eval env
    eval_env.obs_rms = train_env.obs_rms

    # # Stop training if no model improvement after 10 evaluations
    # stop_callback = StopTrainingOnNoModelImprovement(
    #     max_no_improvement_evals=200,
    #     min_evals=1,
    #     verbose=1
    # )

    best_ckpt = SaveBestWithStats(
        eval_env,
        vecnormalize=train_env,
        number=2,
        best_model_save_path=f"pymunk_experiments/gripper2/discrete_control/RL_transfer_experiments/2/models/ppo_pymunk_gripper_best",
        n_eval_episodes=5,
        eval_freq=20_000,
        deterministic=True,
        render=False,
        verbose=0,
        # callback_after_eval=stop_callback
    )

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=500000, render=True, verbose=0, deterministic=False)

    custom_logger = CustomLoggingCallback()
    param_log = ParameterLogger("pymunk_experiments/gripper2/discrete_control/RL_transfer_experiments/ppo_gripper_tensorboard/", log_freq=10000)

    model.learn(total_timesteps=3000000, callback=[eval_callback, best_ckpt, custom_logger, param_log])

    print("Training complete and model saved")

    train_env.close()
    eval_env.close()



if __name__ == "__main__":

    vertex = [(-30, -30), (30, -30), (30, 30), (-30, 30)]

    design_vector = ((200, 120, 120, 120, 120))  # Example design vector

    train_from_scratch(vertex, design_vector)

    print("Training from scratch complete.")

    design_vector = ((200, 100, 130, 110, 125))  # Example design vector for similar gripper

    train_from_similar(vertex, design_vector)






