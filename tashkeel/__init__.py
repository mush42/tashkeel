# coding: utf-8

import os
import numpy as np
from .shakkala import Shakkala


SHAKKALA_INSTANCE = None


def tashkeel(input_text):
    global SHAKKALA_INSTANCE
    if SHAKKALA_INSTANCE is None:
        SHAKKALA_INSTANCE = Shakkala()
    sh = SHAKKALA_INSTANCE
    input_int = sh.prepare_input(input_text).astype(np.float32)
    logits = sh.model.run(None, {'embedding_7_input': input_int})[0][0]
    predicted_harakat = sh.logits_to_text(logits)
    return sh.get_final_text(input_text, predicted_harakat)
