import tk
from tk.jaxline import base_config
from tk.models.gpt2 import GPTConfig


def get_config(debug: str = '0'):
    # NB, I think with ml_collections param has to be a string
    truthy = ('1', 'T', 'True')
    assert debug in truthy + ('0', 'F', 'False')

    def m(default_value, debug_value):
        """Convenience for debug runs."""
        return debug_value if (debug in truthy) else default_value

    cfg = base_config.get_base_config()
    from tk.arc.expt import gpt2
    cfg.experiment_class = gpt2.Experiment
    cfg.experiment_kwargs = dict(
        lr=1e-4,
        batch_size=4,
        model_config=GPTConfig(
            vocab_size=-1,
            block_size=2048,
            num_heads=8,
            num_layers=6,
            num_embeds=256,
            use_bias=True,
            dtype='float32',
        ),
    )

    cfg.best_model_eval_metric = 'loss'
    cfg.best_model_eval_metric_higher_is_better = False
    cfg.training_steps = m(int(5e5), 16)
    cfg.log_train_data_interval = m(60, 1)
    cfg.log_tensors_interval = m(60, 1)
    cfg.save_checkpoint_interval = m(300, 1)
    cfg.train_checkpoint_all_hosts = False
    cfg.checkpoint_dir = tk.datadir / 'outputs' / 'arc-ckpt'
    # can set as --config.restore_path during startup 
    # NB, needs to be string not e.g. None
    cfg.restore_path = '' #tk.datadir / 'outputs' / 'arc-ckpt' / 'models' / 'latest' / 'step...'

    return cfg