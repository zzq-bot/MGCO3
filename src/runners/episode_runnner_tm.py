from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from copy import deepcopy

class EpisodeRunnerTM:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, tm_index):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.tm_index = tm_index

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, t_env_tm0, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=t_env_tm0, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=t_env_tm0, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = f"test_tm_{self.tm_index}_" if test_mode else f"tm_{self.tm_index}_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # if episode_return > 0:
        #     term_step = self.batch['terminated'].flatten().nonzero().item()
        #     term_state = self.batch['state'][0][term_step + 1].flatten()
        #     do_print = self.args.evaluate
        #     if (term_state[-1] == 0 and term_state[-2] == 1) or (term_state[-1] == 1 and term_state[-2] == 0):
        #         cur_stats["food_A"] = cur_stats.get("food_A", 0) + 1
        #         if do_print:
        #             print('A', end = ' ')
        #     if (term_state[-1] == 4 and term_state[-2] == 0) or (term_state[-1] == 5 and term_state[-2] == 1):
        #         cur_stats["food_B"] = cur_stats.get("food_B", 0) + 1
        #         if do_print:
        #             print('B', end = ' ')
        #     if (term_state[-1] == 0 and term_state[-2] == 4) or (term_state[-1] == 1 and term_state[-2] == 5):
        #         cur_stats["food_C"] = cur_stats.get("food_C", 0) + 1
        #         if do_print:
        #             print('C', end = ' ')
        #     if (term_state[-1] == 4 and term_state[-2] == 5) or (term_state[-1] == 5 and term_state[-2] == 4):
        #         cur_stats["food_D"] = cur_stats.get("food_D", 0) + 1
        #         if do_print:
        #             print('D', end = ' ')
        # elif episode_return == 0:
        #     cur_stats["food_X"] = cur_stats.get("food_X", 0) + 1
            
        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
            
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            cur_returns_ = deepcopy(cur_returns)
            cur_stats_ = deepcopy(cur_stats)
            self._log(cur_returns, cur_stats, log_prefix, t_env_tm0)
            # cur_stats["food_X"] = 0
            # cur_stats["food_A"] = 0
            # cur_stats["food_B"] = 0
            # cur_stats["food_C"] = 0
            # cur_stats["food_D"] = 0
            return cur_returns_
        elif (not test_mode) and t_env_tm0 - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix, t_env_tm0)
            # if hasattr(self.mac.action_selector, "epsilon"):
            #     self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = t_env_tm0
            # cur_stats["food_X"] = 0
            # cur_stats["food_A"] = 0
            # cur_stats["food_B"] = 0
            # cur_stats["food_C"] = 0
            # cur_stats["food_D"] = 0

        return self.batch

    def _log(self, returns, stats, prefix, t_env_tm0):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), t_env_tm0)
        self.logger.log_stat(prefix + "return_std", np.std(returns), t_env_tm0)
        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], t_env_tm0)
        stats.clear()
