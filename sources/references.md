# References

- WikiANN: https://huggingface.co/datasets/unimelb-nlp/wikiann
- BERT token classification: https://huggingface.co/docs/transformers/tasks/token_classification
- NLTK Movie Reviews: https://www.nltk.org/nltk_data/
- Gensim Word2Vec: https://radimrehurek.com/gensim/models/word2vec.html
- Padova educational data: https://github.com/ferrazzipietro/dataset-nlp-math
- Padova GPT typo exercise data: https://github.com/elenipapadopulos/NLP_LAB3_Datasets
- HSE NLP teaching materials, 2026 branch: https://github.com/ashaba1in/hse-nlp/tree/2026
- HSE course pages: http://wiki.cs.hse.ru/Современный_NLP_и_большие_языковые_модели_26 and http://wiki.cs.hse.ru/Большие_языковые_модели_25_26

## Methods used in the advanced notebooks

- Transformer: https://arxiv.org/abs/1706.03762
- BERT: https://arxiv.org/abs/1810.04805
- RoPE: https://arxiv.org/abs/2104.09864
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- Reciprocal Rank Fusion: https://dl.acm.org/doi/10.1145/1571941.1572114
- Sentence Transformers: https://www.sbert.net/
- Hugging Face PEFT: https://huggingface.co/docs/peft/

The course materials define the topic progression. Equations and implementations in this repository were written independently and are validated by local tests.

## ML4HD translation models

The uploaded Lab_seq2seq and Lab_transformers examples informed projects 14 and 15. GRU attention and encoder-decoder Transformer code were reimplemented in PyTorch. Original course weights are not used. The default grammar corpus is newly constructed for a controlled experiment; it is not the original course parallel corpus.

The source and target vocabularies are fitted on training text only. Packed recurrent inputs, source attention masks and causal target masks are validated explicitly. Primary references: [Bahdanau attention](https://arxiv.org/abs/1409.0473), [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html).
