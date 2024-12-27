from tk.jaxline import base_config


def get_config(debug: bool = False):

    def m(default_value, debug_value):
        """Convenience for debug runs."""
        return debug_value if debug else default_value

    cfg = base_config.get_base_config()
    from tk.arc.expt import gpt2
    cfg.experiment_class = gpt2.Experiment
    cfg.experiment_kwargs = dict(

    )

    cfg.training_steps = int(5e5)
    cfg.log_train_data_interval = 60
    cfg.log_tensors_interval = 60
    cfg.save_checkpoint_interval = 300
    cfg.train_checkpoint_all_hosts = False
    cfg.checkpoint_dir = '/tmp/tk/'
    cfg.eval_specific_checkpoint_dir = ''
    cfg.restore_path = ''

    return cfg