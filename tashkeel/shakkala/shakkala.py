#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
License
-------
    The MIT License (MIT)

    Copyright (c) 2017 Tashkel Project

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

Created on Sat Dec 16 22:46:28 2017

@author: Ahmad Barqawi
"""
from . import helper
import os
import onnxruntime as ort
import numpy as np


MAX_CHARS = 315
ORT_PROVIDERS = ["CPUExecutionProvider"]
THIS_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
DATA_DIRECTORY = os.path.join(THIS_DIRECTORY, "data")
DICT_DIRECTORY = os.path.join(DATA_DIRECTORY, "dictionary")
MODEL_PATH = os.path.join(DATA_DIRECTORY, "model", "model.onnx")


class Shakkala:
    def __init__(self):
        self.model = ort.InferenceSession(MODEL_PATH, providers=ORT_PROVIDERS)
        input_vocab_to_int = helper.load_binary("input_vocab_to_int", DICT_DIRECTORY)
        output_int_to_vocab = helper.load_binary("output_int_to_vocab", DICT_DIRECTORY)

        self.dictionary = {
            "input_vocab_to_int": input_vocab_to_int,
            "output_int_to_vocab": output_int_to_vocab,
        }

    def prepare_input(self, input_sent):
        assert (
            input_sent != None and len(input_sent) < MAX_CHARS
        ), "max length for input_sent should be {} characters, you can split the sentence into multiple sentecens and call the function".format(
            MAX_CHARS
        )
        input_sent = [input_sent]
        return self.__preprocess(input_sent)

    def __preprocess(self, input_sent):
        input_vocab_to_int = self.dictionary["input_vocab_to_int"]
        input_letters_ids = [
            [input_vocab_to_int.get(ch, input_vocab_to_int["<UNK>"]) for ch in sent]
            for sent in input_sent
        ]
        input_letters_ids = self.__pad_size(input_letters_ids, MAX_CHARS)
        return input_letters_ids

    def logits_to_text(self, logits):
        text = []
        for prediction in np.argmax(logits, 1):
            if self.dictionary["output_int_to_vocab"][prediction] == "<PAD>":
                continue
            text.append(self.dictionary["output_int_to_vocab"][prediction])
        return text

    def get_final_text(self, input_sent, output_sent):
        return helper.combine_text_with_harakat(input_sent, output_sent)

    def clean_harakat(self, input_sent):
        return helper.clear_tashkel(input_sent)

    def __pad_size(self, x, length=None):
        return helper.pad_sequences(x, maxlen=length, padding="post")
