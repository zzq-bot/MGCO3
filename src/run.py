import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import json
from copy import deepcopy
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    
    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        tb_exp_direc = os.path.join(results_save_dir, 'tb_logs')
        logger.setup_tb(tb_exp_direc)

        config_str = json.dumps(vars(args), indent=4)
        with open(os.path.join(results_save_dir, "config.json"), "w") as f:
            f.write(config_str)

    
    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    raise NotImplementedError
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):
    ############
    # Init ego #
    ############
    # Init runner so we can get env info
    runner_ego = r_REGISTRY[args.runner_ego](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(args, k, v)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer_ego = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac_ego = mac_REGISTRY[args.mac](buffer_ego.scheme, groups, args)

    # Give runner the scheme
    runner_ego.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac_ego=mac_ego)

    # Learner
    learner_ego = le_REGISTRY[args.learner](mac_ego, buffer_ego.scheme, logger, args)

    if args.use_cuda:
        learner_ego.cuda()

    if args.checkpoint_path_ego != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path_ego):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path_ego))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path_ego):
            full_name = os.path.join(args.checkpoint_path_ego, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path_ego, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner_ego.load_models(model_path)
        runner_ego.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner_ego)
            return

    ##################
    # Init Teammates #
    ##################
    tm2runner, tm2buffer, tm2learner, tm2mac = {}, {}, {}, {}
    for tm in range(args.n_population):
    # Init runner so we can get env info
        runner_tm = r_REGISTRY[args.runner_tm](args=args, logger=logger)
        mac_tm = deepcopy(mac_ego)
        # Give runner the scheme
        runner_tm.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac_tm, tm_index=tm)

        tm2runner[tm] = runner_tm
        tm2buffer[tm] = deepcopy(buffer_ego)
        tm2learner[tm] = deepcopy(learner_ego)
        tm2mac[tm] = mac_tm

    if args.checkpoint_path_tm != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path_tm):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path_tm))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path_tm):
            full_name = os.path.join(args.checkpoint_path_tm, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path_tm, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        for tm, mac in tm2mac.items():
            mac.load_models(model_path)

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, tm2runner)
            return
    ################
    # Pretrain Ego #
    ################
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning pretraining ego for {} timesteps".format(args.t_pretrain_ego))

    for _ in range(args.test_nepisode):
        runner_ego.run(mac_ego, test_mode=True, test_mode_ego=True, test_mode_tm=True, pretrain=True)

    while runner_ego.t_env <= args.t_pretrain_ego:

        # Run for a whole episode at a time
        episode_batch = runner_ego.run(mac_ego, test_mode=False, test_mode_ego=False, test_mode_tm=False, pretrain=True)
        buffer_ego.insert_episode_batch(episode_batch)

        if buffer_ego.can_sample(args.batch_size):
            episode_sample = buffer_ego.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner_ego.train(episode_sample, runner_ego.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner_ego.batch_size)
        if (runner_ego.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("pretrain t_env: {} / {}".format(runner_ego.t_env, args.t_pretrain_ego))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner_ego.t_env, args.t_pretrain_ego), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner_ego.t_env
            for _ in range(n_test_runs):
                runner_ego.run(mac_ego, test_mode=True, test_mode_ego=True, test_mode_tm=True, pretrain=True)

        if args.save_model and (runner_ego.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner_ego.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner_ego.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner_ego.save_models(save_path)

        episode += args.batch_size_run

        if (runner_ego.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner_ego.t_env)
            logger.print_recent_stats()
            last_log_T = runner_ego.t_env

    model_save_time = runner_ego.t_env
    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner_ego.t_env))
    #"results/models/{}".format(unique_token)
    os.makedirs(save_path, exist_ok=True)
    logger.console_logger.info("Saving models to {}".format(save_path))

    # learner should handle saving/loading -- delegate actor save/load to mac,
    # use appropriate filenames to do critics, optimizer states
    learner_ego.save_models(save_path)

    #############
    # Iteration #
    #############
    for iter in range(args.max_iteration):
        ###################
        # Train teammates #
        ###################
        #############
        # Train ego #
        #############
        pass

    runner_ego.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
