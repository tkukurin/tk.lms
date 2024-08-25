"""Wrapper around annoying metrics and other logging stuff.
"""
from pathlib import Path
from typing import Callable
from loguru import logger


def get_logger_write_method(output_dir: Path) -> Callable:
    """I switched to aim from tensorboard for now.

    TODO update things to only use one method or sth.
    For now just leave the hackish method around.
    
    Also see e.g. [convert].
    
    [convert]: https://aimstack.readthedocs.io/en/latest/quick_start/convert_data.html#show-tensorboard-logs-in-aim
    """
    run = None

    try:
        import aim
        import tk
        run = aim.Run(repo=tk.rootdir)
        # TODO at some point other setup etc 
        # https://aimstack.readthedocs.io/en/latest/using/artifacts.html
        # run.set_artifacts_uri("s3://aim/artifacts/")
    except ImportError as ie:
        logger.warning(
            f"Unable to display metrics through aim because some package are not installed: {ie}"
        )

    if run is None:
        try:
            from tensorboardX import SummaryWriter
            # requires tensorflow import ?!
            # from flax.metrics.tensorboard import SummaryWriter
            run = SummaryWriter(log_dir=Path(output_dir))
        except ImportError as ie:
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
            return (lambda *a, **kw: logger.info(f"{a=} {kw=}"))

    return _get_track_method(run)


def _get_track_method(summary_writer):
    try: 
        import aim
        if isinstance(summary_writer, aim.Run):
            # order consistent w tensorflow
            add = lambda tag, val, step: summary_writer.track(
                val, name=tag, step=step
            )
    except ImportError:
        if hasattr(summary_writer, 'add_scalar'):
            add = summary_writer.add_scalar
        else:
            add = summary_writer.scalar

    return add
