from argparse import Namespace
from models.bert import BertClf
from models.mmbt import MultimodalBertClf

def get_model(hparams: Namespace = None):
    """
    Get model
    Args:
        hparams: config
    Returns:
    """
    if hparams.model == "bert":
        model = BertClf(hparams)
    elif hparams.model == "mmbt":
        model = MultimodalBertClf(hparams)
    else:
        raise ValueError(f'Specified model ({hparams.model}) not implemented')

    return model