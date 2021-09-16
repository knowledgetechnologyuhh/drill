import os
import time

# Enable comet logging
try:
    from comet_ml import Experiment, Optimizer

    online_logging = True
    COMET_KEY = os.environ["COMET_KEY"]
    COMET_WORKSPACE = os.environ["COMET_WORKSPACE"]
    COMET_PROJ_NAME = os.environ["COMET_PROJ_NAME"]
except:
    online_logging = False

import logging
import os
import random
from argparse import ArgumentParser
import json

import numpy as np
import torch

import datasets.utils
import models.utils

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('General-Log')

if __name__ == '__main__':
    # Dataset ordering presets
    dataset_order_mapping = {
        1: ['Yelp', 'AGNews', 'DBPedia', 'Amazon', 'Yahoo'],
        2: ['DBPedia', 'Yahoo', 'AGNews', 'Amazon', 'Yelp'],
        3: ['Yelp', 'Yahoo', 'Amazon', 'DBPedia', 'AGNews'],
        4: ['AGNews', 'Yelp', 'Amazon', 'Yahoo', 'DBPedia']
    }

    # Parse the configuration file if provided
    parser = ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='Configuration file')
    parser.add_argument('--cfg_tag', default='default', type=str, help='Configuration tag')
    parser.add_argument('--hyperparam_cfg_file', type=str,
                        help='Hyperparameter optimization configuration file (Comet ML)')
    args, _ = parser.parse_known_args()
    if args.cfg_file:
        with open(args.cfg_file) as f:
            config = json.load(f)
        defaults = config[args.cfg_tag]
    else:
        defaults = {}
    if args.hyperparam_cfg_file:
        hyperparam_opt = Optimizer(args.hyperparam_cfg_file, api_key=COMET_KEY)
        experiments = lambda: hyperparam_opt.get_experiments(project_name=COMET_PROJ_NAME, workspace=COMET_WORKSPACE)
    elif online_logging:
        hyperparam_opt = None
        experiments = lambda: [Experiment(api_key=COMET_KEY, project_name=COMET_PROJ_NAME, workspace=COMET_WORKSPACE)]
    else:
        hyperparam_opt = None
        experiments = lambda: [None]

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--balance_replay', help='Overrides the write probabilities to balance the replay buffer when '
                                                 '--prog_reduction or --prog_expansion in place', action='store_true')
    parser.add_argument('--batch_size', type=int, help='Batch size for training (b)', default=16)
    parser.add_argument('--cleanup_freq', type=int, help='Frequency of pruning old nodes by a quarter of the current '
                                                         'size to reduce memory overhead', default=1000)
    parser.add_argument('--datasets', '-d', nargs='+', type=str, help='Datasets used for training',
                        default=['YelpShort'])
    parser.add_argument('--del_freq', type=int, help='Frequency of node pruning', default=10)
    parser.add_argument('--dim', type=int, help='Dimension of growing memory and language model output', default=768)
    parser.add_argument('--epochs', type=int, help='Number of epochs (used for multitask setup)', default=1)
    parser.add_argument('--gpu', type=int, help='The ID of the GPU to be used')
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=1e-3)
    parser.add_argument('--learner', type=str, help='Learner method', default='replay')
    parser.add_argument('--log_freq', type=int, help='Logging frequency of learning metrics', default=10)
    parser.add_argument('--lr', type=float, help='Learning rate', default=3e-5)
    parser.add_argument('--max_age', type=int, help='Maximum age of a node connecting edge', default=10)
    parser.add_argument('--max_len', type=int, help='Maximum sequence length for the transformer input', default=448)
    parser.add_argument('--memory', type=str, help='Memory model', default='NONE')
    parser.add_argument('--memory_iter', type=int, help='Memory iterations within inner loop (defaults to batch_size)')
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--order', '-o', type=int, help='Order of datasets (overrides datasets)')
    parser.add_argument('--prog_reduction', help='Progressive reduction of train dataset size to enforce data '
                                                 'imbalance (reduce_train[0]/2**ds_idx)', action='store_true')
    parser.add_argument('--prog_expansion', help='Progressive expansion of train dataset size to enforce data '
                                                 'imbalance (reduce_train[0]/2**(N_ds-ds_idx))', action='store_true')
    parser.add_argument('--pull_factor', type=int, help='Pulling strength between neighboring nodes (1/pull_factor)',
                        default=50)
    parser.add_argument('--reduce_test', nargs='+', type=int, help='Maximum number of test samples per dataset',
                        default=[7600])
    parser.add_argument('--reduce_train', nargs='+', type=int, help='Maximum number of train samples per dataset',
                        default=[115000])
    parser.add_argument('--replay_every', type=int, help='Number of data points between replay (R_I)', default=9600)
    parser.add_argument('--replay_rate', type=float, help='Replay rate from memory (r)', default=0.01)
    parser.add_argument('--seed', type=int, help='Random state for reproducible output', default=42)
    parser.add_argument('--supervised_sampling', help='Enables supervised sampling of SOINN+ BMUs (defaults to closest '
                                                      'neighbor sampling)', action='store_true')
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates / buffer size (m)', default=5)
    parser.add_argument('--write_probs', nargs='+', type=float, help='Write probabilities for buffer memory (providing '
                                                                     'a single value broadcasts same prob. for all datasets)',
                        default=[1.0])

    parser.set_defaults(**defaults)
    args, _ = parser.parse_known_args()

    # Set random state
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set base path (current working directory)
    base_path = os.getcwd()

    # Set processing device
    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    for experiment in experiments():
        if args.order:
            vars(args).update(datasets=dataset_order_mapping[args.order])
        if hyperparam_opt is not None:
            vars(args).update(**experiment.params)

        if len(args.reduce_test) == 1:
            args.reduce_test = [args.reduce_test[0]] * len(args.datasets)
        if len(args.reduce_train) == 1:
            args.reduce_train = [args.reduce_train[0]] * len(args.datasets)
        if len(args.write_probs) == 1:
            args.write_probs = [args.write_probs[0]] * len(args.datasets)

        if args.prog_reduction and args.prog_expansion:
            raise ValueError("Cannot apply progressive reduction and expansion. Choose 1 only")
        if args.prog_reduction:
            args.reduce_train = [args.reduce_train[0]] * len(args.datasets)
            args.write_probs = [args.write_probs[0]] * len(args.datasets) if args.balance_replay else args.write_probs
            for ds_idx, reduce_train in enumerate(args.reduce_train):
                args.reduce_train[ds_idx] //= 2 ** ds_idx
                if args.balance_replay:
                    args.write_probs[ds_idx] /= 2 ** (len(args.datasets) - 1 - ds_idx)
        elif args.prog_expansion:
            args.reduce_train = [args.reduce_train[0]] * len(args.datasets)
            args.write_probs = [args.write_probs[0]] * len(args.datasets) if args.balance_replay else args.write_probs
            for ds_idx, reduce_train in enumerate(args.reduce_train):
                args.reduce_train[ds_idx] //= 2 ** (len(args.datasets) - 1 - ds_idx)
                if args.balance_replay:
                    args.write_probs[ds_idx] /= 2 ** ds_idx

        try:
            # Initialize online logger
            if online_logging:
                experiment.add_tag(str(args.order))
                experiment.set_name(args.learner)
                experiment.log_parameters(vars(args))
                vars(args).update(online_logger=experiment)

            logger.info('Using configuration: {}'.format(vars(args)))

            # Load train and test datasets
            logger.info('Loading all datasets...')
            train_datasets, test_datasets, n_classes = [], [], 0
            for ds_idx, dataset in enumerate(args.datasets):
                train_dataset, test_dataset = datasets.utils.load_dataset(base_path, dataset, args.reduce_train[ds_idx],
                                                                          args.reduce_test[ds_idx])
                logger.info('Loaded {}.'.format(train_dataset.__class__.__name__))
                n_classes += train_dataset.n_classes
                datasets.utils.offset_labels(train_dataset)
                datasets.utils.offset_labels(test_dataset)
                train_datasets.append(train_dataset)
                test_datasets.append(test_dataset)
            n_classes = n_classes - 5 if args.order else n_classes
            logger.info('Successfully loaded all datasets.')

            # Initialize model
            logger.info('Initializing model instance...')
            model = models.utils.init_model(device, n_classes, **vars(args))
            logger.info('Successfully initialized {} instance.'.format(model.__class__.__name__))

            # Train model
            logger.info('----------Training starts here----------')
            concat_dataset = model.training(train_datasets, **vars(args))
            logger.info('-----------Training completed-----------')

            # Test model
            logger.info('----------Testing starts here----------')
            model.testing(test_datasets, **vars(args))
            logger.info('-----------Testing completed-----------')

            if online_logging:
                experiment.log_metric("loss", experiment.get_metric("train_query__loss"))
                time.sleep(10)
                experiment.clean()
                time.sleep(1)

        except ConnectionError:
            time.sleep(10)
