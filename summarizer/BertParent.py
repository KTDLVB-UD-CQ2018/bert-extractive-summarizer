from transformers import *
import logging
import torch
import numpy as np
from numpy import ndarray
from typing import List
from vncorenlp import VnCoreNLP

logging.basicConfig(level=logging.WARNING)


class BertParent(object):

    MODELS = {
        'vinai/phobert-base': (None, None)
    }

    def __init__(
        self,
        model: str,
        custom_model: PreTrainedModel=None,
        custom_tokenizer: PreTrainedTokenizer=None
    ):
        """
        :param model: Model is the string path for the bert weights. If given a keyword, the s3 path will be used
        :param custom_model: This is optional if a custom bert model is used
        :param custom_tokenizer: Place to use custom tokenizer
        """

        self.model = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.model.eval()
        self.rdrsegmenter = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

    def tokenize_input(self, text: str) -> torch.tensor:
        """
        Tokenizes the text input.
        :param text: Text to tokenize
        :return: Returns a torch tensor
        """
        tokenized_text = self.tokenizer.tokenize(text)#self.rdrsegmenter.tokenize(text)[0]
        #print(f'tokenized text: {tokenized_text}')
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens])

    def extract_embeddings(
        self,
        text: str,
        hidden: int=-2,
        squeeze: bool=False,
        reduce_option: str ='mean'
    ) -> ndarray:
        #print(f'raw text: {text}')
        tokens_tensor = self.tokenize_input(text)
        pooled, hidden_states = self.model(tokens_tensor)[-2:]

        if -1 > hidden > -12:

            if reduce_option == 'max':
                pooled = hidden_states[hidden].max(dim=1)[0]

            elif reduce_option == 'median':
                pooled = hidden_states[hidden].median(dim=1)[0]

            else:
                pooled = hidden_states[hidden].mean(dim=1)

        if squeeze:
            return pooled.detach().numpy().squeeze()

        return pooled

    def create_matrix(
        self,
        content: List[str],
        hidden: int=-2,
        reduce_option: str = 'mean'
    ) -> ndarray:
        return np.asarray([
            np.squeeze(self.extract_embeddings(t, hidden=hidden, reduce_option=reduce_option).data.numpy())
            for t in content
        ])

    def __call__(
        self,
        content: List[str],
        hidden: int= -2,
        reduce_option: str = 'mean'
    ) -> ndarray:
        return self.create_matrix(content, hidden, reduce_option)
