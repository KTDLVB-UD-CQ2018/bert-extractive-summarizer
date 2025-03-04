from summarizer.BertParent import BertParent
from typing import List
from summarizer.ClusterFeatures import ClusterFeatures
from abc import abstractmethod
import neuralcoref
from spacy.lang.vi import Vietnamese
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from vncorenlp import VnCoreNLP


class ModelProcessor(object):

    def __init__(
        self,
        model='vinai/phobert-base',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int=-2,
        reduce_option: str = 'mean',
        greedyness: float=0.45,
        language=Vietnamese,
        random_state: int = 12345
    ):
        np.random.seed(random_state)
        self.model = BertParent(model, custom_model, custom_tokenizer)
        self.hidden = hidden
        self.reduce_option = reduce_option
        self.nlp = language()
        self.random_state = random_state
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.rdrsegmenter = VnCoreNLP("/content/bert-extractive-summarizer/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
        neuralcoref.add_to_pipe(self.nlp, greedyness=greedyness)

    def process_content_sentences(self, body: str, min_length=40, max_length=600) -> List[str]:
        doc = []
        sentences = self.rdrsegmenter.tokenize(body) 
        for sentence in sentences:
            sent = ''
            for word in sentence:
                sent = sent + ' ' + word
            doc.append(sent)
        return [c.strip() for c in doc if max_length > len(c.strip()) > min_length]

    @abstractmethod
    def run_clusters(self, content: List[str], ratio=0.2, algorithm='kmeans', use_first: bool=True) -> List[str]:
        raise NotImplementedError("Must Implement run_clusters")

    def run(
        self,
        body: str,
        ratio: float=0.2,
        min_length: int=40,
        max_length: int=600,
        use_first: bool=True,
        algorithm='kmeans'
    ) -> str:
        #print(f'content: {body}')
        #raise Exception()
        sentences = self.process_content_sentences(body, min_length, max_length)

        if sentences:
            sentences = self.run_clusters(sentences, ratio, algorithm, use_first)

        return sentences#' '.join(sentences)

    def __call__(self, body: str, ratio: float=0.2, min_length: int=40, max_length: int=600,
                 use_first: bool=True, algorithm='kmeans') -> str:
        return self.run(body, ratio, min_length, max_length)


class SingleModel(ModelProcessor):
    """
    Deprecated for naming sake.
    """

    def __init__(
        self,
        model='vinai/phobert-base',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int=-2,
        reduce_option: str = 'mean',
        greedyness: float=0.45,
        language=Vietnamese,
        random_state: int=12345
    ):
        super(SingleModel, self).__init__(model, custom_model, custom_tokenizer, hidden, reduce_option,
                                          greedyness, language=language, random_state=random_state)

    def run_clusters(self, content: List[str], ratio=0.2, algorithm='kmeans', use_first: bool= True) -> List[str]:
        
        hidden = self.model(content, self.hidden, self.reduce_option)
        hidden_args = ClusterFeatures(hidden, algorithm, random_state=self.random_state).cluster(ratio)

        if use_first:
            if hidden_args[0] != 0:
                hidden_args.insert(0,0)
        #print('Cluster content')
        #for c in content:
        #    print(f'{c}')
        return [content[j] for j in hidden_args]


class Summarizer(SingleModel):

    def __init__(
        self,
        model='vinai/phobert-base',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int=-2,
        reduce_option: str = 'mean',
        greedyness: float=0.45,
        language=Vietnamese,
        random_state: int=12345
    ):
        super(Summarizer, self).__init__(model, custom_model, custom_tokenizer, hidden, reduce_option, greedyness, language, random_state)
