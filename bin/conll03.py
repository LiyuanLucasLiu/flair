from flair.data import NLPTaskDataFetcher, TaggedCorpus, NLPTask
from flair.embeddings import TextEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings
from typing import List
import torch

# if args.gpu:
torch.cuda.set_device(7)
# device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.CONLL_03)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TextEmbeddings] = [

    # GloVe embeddings
    WordEmbeddings('glove')
    ,
    # contextual string embeddings, forward
    CharLMEmbeddings('news-forward')
    ,
    # contextual string embeddings, backward
    CharLMEmbeddings('news-backward')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.tagging_model import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
                                        
if torch.cuda.is_available():
    tagger = tagger.cuda()

# initialize trainer
from flair.trainer import TagTrain

trainer: TagTrain = TagTrain(tagger, corpus, test_mode=False)

trainer.train('resources/taggers/example-ner', mini_batch_size=32, max_epochs=150, save_model=False,
              train_with_dev=False, anneal_mode=True)
