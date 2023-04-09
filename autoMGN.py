from openbox import space as sp
from openbox import Optimizer

from train import main


def get_configspace():
    space = sp.Space()
    message_passing_step = sp.Int("message_passing_step", 2, 8)
    # n_estimators = sp.Int("n_estimators", 100, 1000, default_value=500, q=50)
    # num_leaves = sp.Int("num_leaves", 31, 2047, default_value=128)
    # max_depth = sp.Constant('max_depth', 15)
    learning_rate = sp.Real("learning_rate", 1e-4, 0.3, default_value=0.1, log=True)
    gamma = sp.Real("gamma", 0.65, 0.95)
    # min_child_samples = sp.Int("min_child_samples", 5, 30, default_value=20)
    # subsample = sp.Real("subsample", 0.7, 1, default_value=1, q=0.1)
    # colsample_bytree = sp.Real("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
    space.add_variables([message_passing_step, learning_rate, gamma])
    return space


def objective_function(config: sp.Configuration):
    params = config.get_dictionary().copy()
    # params['n_jobs'] = 2
    # params['random_state'] = 47
    # train.main()
    loss = main(**params)  # minimize
    return dict(objectives=[loss])


if __name__ == '__main__':
    # Run
    opt = Optimizer(
        objective_function,
        get_configspace(),
        num_objectives=1,
        num_constraints=0,
        max_runs=4,
        surrogate_type='gp',
        task_id='so_hpo',
        # Have a try on the new HTML visualization feature!
        visualization='basic',   # or 'basic'. For 'advanced', run 'pip install "openbox[extra]"' first
        auto_open_html=True,        # open the visualization page in your browser automatically
    )
    history = opt.run()
