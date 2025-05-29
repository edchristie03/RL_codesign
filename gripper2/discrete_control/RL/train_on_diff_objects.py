from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import Environment


shape_vertices = {
    "Circle":          [],
    "Square":          [(-30, -30), (30, -30), (30, 30), (-30, 30)],
    "RA triangle":     [(-30, -30), (30, -30), (30, 30)],
    "Equi Triangle":   [(-30, -30), (30, -30), (0, 30)]}
#     "Half Trapezoid":  [(-30, -30), (30, -30), (0, 30), (-30, 30)],
#     "Tall Rectangle":  [(-10, -30), (0, -30), (0, 30), (-10, 30)],
#     "Wide Rectangle":  [(-80, -30), (80, -30), (80, 0), (-80, 0)],
# }


if __name__ == "__main__":

    for shape_name, vertex in shape_vertices.items():
        N_ENVS = 8  # Number of parallel environments

        # Define the policy network architecture
        policy_kwargs = {'net_arch': [256, 256], "log_std_init": 2}

        def make_env(vertex, rank, render=False):
            """
            Factory that creates a *fresh* environment in its own process.
            `rank` is only used if you want per‑worker seeding or logging.
            """

            def _init():
                env = Environment(vertex, training=True, render_mode="human" if render else None)
                env = Monitor(env)  # keeps episode stats
                return env

            return _init

        class SaveBestWithStats(EvalCallback):
            def __init__(self, *args, vecnormalize, **kwargs):
                super().__init__(*args, **kwargs)
                self.vecnormalize = vecnormalize  # the training wrapper

            def _on_step(self) -> bool:
                # run the usual evaluation logic
                old_reward = self.best_mean_reward
                continue_training = super()._on_step()

                if self.best_mean_reward > old_reward:  # new best just saved
                    self.vecnormalize.save(f"normalise_stats/vecnormalize_stats_best_{shape_name}.pkl")

                return continue_training

        # Training envs (headless, parallel)
        train_env = SubprocVecEnv([make_env(vertex, i) for i in range(N_ENVS)], start_method="spawn")
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

        # Env for saving best model
        eval_env_fast = DummyVecEnv([make_env(vertex, 0, render=False)])
        eval_env_fast = VecNormalize(eval_env_fast, norm_obs=True, norm_reward=False, training=False)
        eval_env_fast.obs_rms = train_env.obs_rms

        best_ckpt = SaveBestWithStats(
            eval_env_fast,
            vecnormalize=train_env,
            best_model_save_path=f"models/{shape_name}",
            n_eval_episodes=10,
            eval_freq=20_000,
            deterministic=True,
            render=False,
            verbose=0,
        )

        # Evaluation env (still a single window so you can watch)
        eval_env = DummyVecEnv([make_env(vertex, 0, render=True)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = train_env.obs_rms  # share running stats

        # Make callback to run 1 episode every eval_freq steps
        eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=1000000, render=True, verbose=0,
                                     deterministic=True)

        # Instantiate PPO on the train_env, pass the callback to learn()
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=256,  # 256 × 8 = 2048 steps / update
            batch_size=512,  # must divide N_ENVS × N_STEPS
            verbose=0,
            tensorboard_log="./ppo_gripper_tensorboard/",
            policy_kwargs=policy_kwargs,
            ent_coef=0.05,
            learning_rate=1e-3,
        )

        model.learn(total_timesteps=10000000, callback=[eval_callback, best_ckpt])
        print(f"Training complete and model saved for {shape_name}")

        train_env.close()
        eval_env.close()
