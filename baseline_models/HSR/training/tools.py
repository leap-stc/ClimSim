import sys
import time
import random
import numpy as np
from copy import deepcopy
import pickle


"""
Contains methods for miscellaneous tasks, not specific to the models used.
This includes printing (configuration, progress), and hyperparameter optimization.
"""


def progress(iterable, text=None, inner=None, timed=None):
    """
    Generator for timed for loops with progress bar

    :param iterable, inner: iterable for outer and (optional) inner loop
    :param text: (optional) Task description
    :param timed: [list of (delta, f)] events that are triggered by calling <f> after <delta> seconds have passed
    """
    text = text + ' ' if text is not None else ''
    start = time.time()
    last = start
    if timed is not None:
        last_timed = {item: start for item in timed}

    def handle_events(force=False):
        for (dt, f), lt in last_timed.items():
            if now - lt > dt or force:
                f()
                last_timed[(dt, f)] = now

    # for loop
    if inner is None:
        for i, x in enumerate(iterable):
            now = time.time()
            if i == 0 or i == len(iterable) - 1 or now - last > 0.5:
                last = now
                # Progress percentage at step completion, TBD percentage shortly after step start
                perc = (i + 1) / len(iterable)
                inv_perc = len(iterable) / (i + 0.1)
                sys.stdout.write("\r%s[%.1f %%] - %d / %d - %.1fs [TBD: %.1fs]" %
                                 (text, 100 * perc, i + 1, len(iterable), now - start, (now - start) * (inv_perc - 1)))
                sys.stdout.flush()
            # Call events
            if timed is not None:
                handle_events()
            yield x
    # for loop in for loop
    else:
        for i, x in enumerate(iterable):
            for j, y in enumerate(inner):
                now = time.time()
                if j == 0 or j == len(inner) - 1 or now - last > 0.5:
                    last = now
                    perc = (i * len(inner) + j + 1) / (len(iterable) * len(inner))
                    inv_perc = (len(iterable) * len(inner)) / (i * len(inner) + j + 0.1)
                    sys.stdout.write("\r%s[%.1f %%] - %d / %d (%d / %d) - %.1fs [TBD: %.1fs]" %
                                     (text, 100 * perc, i + 1, len(iterable), j + 1, len(inner),
                                      now - start, (now - start) * (inv_perc - 1)))
                    sys.stdout.flush()
                # Call events
                if timed is not None:
                    handle_events()
                yield x, y

    if timed is not None:
        handle_events(force=True)
    print()


def pprint(dict_, k='Parameters', level=0):
    """
    Recursively pretty-print a dict
    """
    if not type(dict_) == dict:
        print(4 * level * ' ' + '%s: %s' % (k, dict_))
        return
    print(4 * level * ' ' + '%s: {' % k)
    [pprint(v, k=k, level=level + 1) for k, v in dict_.items()]
    print(4 * level * ' ' + '}')


def sample_from_sweep(sweep):
    """
    Create a sample from a sweep configuration.
    Supports continuous {min, max, distribution} and discrete {values} values.
    Defined recursively to support arbitrary depth of the configuration dict.
    """
    # Fixed value
    if not type(sweep) == dict:
        return sweep
    # Sample continuous value:
    if 'distribution' in sweep.keys():
        if sweep['distribution'] == 'log_uniform':
            return np.exp(np.random.uniform(low=np.log(sweep['min']), high=np.log(sweep['max'])))
        elif sweep['distribution'] == 'uniform':
            return np.random.uniform(low=sweep['min'], high=sweep['max'])
        else:
            raise ValueError('Unknown distribution')
    # Sample discrete value:
    elif 'values' in sweep.keys():
        return random.choice(sweep['values'])
    # Recurse
    return {k: sample_from_sweep(v) for k, v in sweep.items()}


def hyperparameter_tuning(sweep, train_and_eval, metric=None, runs=10, save_dir=None):
    """
    Rudimentary version of local hyperparameter tuning, with syntax similar to Weights and Biases.
    Supports continuous {min, max, distribution} and discrete {values} values.
    Does a random search over all parameters, prints metrics, and saves trained models / configuration.

    Inputs:
    -------
    sweep - [nested dict] the sweep configuration file,
            see 'parameters' at https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
    train_and_eval - [callable] takes training parameters / save_path returns a dict with entries {metric: value}
    metric - [str] (optional) metric to minimize, i.e. to sort the runs by
    runs - [int] number of runs
    save_dir - [str] (optional) where to save the trained models
    """
    record = [[], [], []]
    for i in range(runs):
        train_params = sample_from_sweep(sweep)
        print('Run %d:' % i)
        pprint(train_params) 
        record[1] += [i]
        record[2] += [deepcopy(train_params)]
        result = train_and_eval(train_params, save_path=save_dir + '%d.cp' % i if save_dir is not None else None)[0]
        record[0] += [result]
    order = sorted(record[1], key=lambda i: record[0][i][metric])
    record = [[r[i] for i in order] for r in record]

    metrics = record[0][0].keys()
    if 'cvae' in save_dir:
        interests = ['layers', 'latent_dims', 'hidden_dims', 'beta', 'lr', 'weight_decay']
    else:
        interests = ['layers', 'hidden_dims', 'lr', 'gamma']
    print('\n     %s | %s' % (' '.join(['%8s' % x for x in metrics]), ' '.join(['%11s' % x for x in interests])))
    for i, id in enumerate(record[1]):
        print('[%2d] %s | %s' % (
            id, ' '.join(['%8.4g' % v for v in record[0][i].values()]),
            ' '.join(['%11.4g' % (record[2][i][k] if k in record[2][i].keys() else record[2][i]['model_params'][k])
                      for k in interests])))
    with open(save_dir + 'records.pickle', 'wb') as f:
        pickle.dump(record, f)

    # with open('models/cvae_sweep/records.pickle', 'rb') as f:
    #     record = pickle.load(f)
    #     pprint(record[2][0])
