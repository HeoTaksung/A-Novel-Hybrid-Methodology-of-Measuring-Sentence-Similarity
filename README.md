# A-Novel-Hybrid-Methodology-of-Measuring-Sentence-Similarity

Sentence Similarity Evaluation

  * [A Novel Hybrid Methodology of Measuring Sentence Similarity](https://www.mdpi.com/2073-8994/13/8/1442)

  * `Yongmin Yoo`, `Tak-Sung Heo`, `Yeongjoon Park`, `Kyungsun Kim`

-------------------------------------------------

## Dataset

  * [KorSTS](https://github.com/kakaobrain/KorNLUDatasets)

-------------------------------------------------

## Model Structure


<p align="center">
	<img src="https://github.com/HeoTaksung/A-Novel-Hybrid-Methodology-of-Measuring-Sentence-Similarity/blob/main/figure.png" alt="Model" width="50%" height="50%"/>
</p>


  * Save the predicted sentence similarity value through BERT using fine-tuning

  * Extracting word embedding vectors for Sentence 1 and Sentence 2 through fine-tuned BERT

  * Applying cosine similarity to the word embedding vectors of sentence 1 and the word embedding vectors of sentence 2

  * Combining the predicted similarity value through fine-tuned BERT and the value of cosine similarity through alpha weight
     
    * α(similarity value through fine-tuned BERT) + (1-α)(word embedding value thorough cosine similarity)

-------------------------------------------------

## Result

  |    Model    | Deep Learning (Pearson/Spearman)  | Hybrid (Pearson/Spearman)  |
  | :------: | :---: | :-----: |
  |  LSTM               | 0.362511 / 0.344804      | 0.557284 / 0.550898      |
  |  CNN+LSTM              | 0.35528 / 0.334058      | 0.556268 / 0.55126      |
  |  G-CNN             | 0.604134 / 0.577296      | 0.65184 / 0.640162      |
  |  Capsule [(Code)](https://github.com/HeoTaksung/Global-and-Local-Information-Adjustment-for-Semantic-Similarity-Evaluation)          | 0.620881 / 0.599617      | 0.661728 / 0.65319     |
  |  BERT       | 0.838989 / 0.830135      | 0.842797 / 0.834181      |
