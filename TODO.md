## Milestone 1

-   [x] Analyse the datasets BEFORE the merge: Are there duplicate movies with different genres? e.g. Movie A in csv A with genre Action, Adventure and Movie A in csv B with Genre Action, Sci-Fi? If yes, why? Is it a problem? (Filip + Tim)

-   [x] Additionally, are there movies with same name and completely different genres and/or descriptions? Definitly analyse duplicate descriptions, are there any? (Filip + Tim)

-   [x] Movies that have two descriptions and one of them is "Add a plot": need to choose the other one

-   [x] Analyse the dataset AFTER the merge: What is the distribution of genres? How are the co-occurances between genres distributed? Do we have severe class imbalance in genres? What is the distribution in length of the descriptions? (Filip + Tim)

-   [x] Preprocessing: tokenization, "normalization": Stopword removal, lemmatization (Daniel)

-   [x] Save processed data in CoNLL-U format (Daniel/Sebastian)

-   [x] Analyse after preprocessing: Word Count? Are there some description with only a few words? Stuff like this. (Sebastian)

-   [x] add wordcloud

-   [x] Description in README of what has been done (at the end, in the final meeting, which also be kick-up for Milestone 2)

-   [x] MORE COMMENTS (All)

## Milestone 2

#### 20-25.11 Review meeting

#### 29.11 Deadline except for DL-approaches

#### 10.12 Internal Deadline

#### 15.12 Deadline

-   [x] Conllu Dataloader (Daniel)

-   [x] Spitting Data -> train.csv, test.csv, dev.csv (Daniel et.al.)

    -   Stratified Split the data into train, test (90/10), fix and save split to file
    -   Stratified Split Dev Set for experimentation (10% of train), fix and save split to file
    -   Iterative / Combination of Labels
    -   Fixed seed for fitting and evaluation (CV): 42069

    -   https://github.com/trent-b/iterative-stratification
    -   http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html

-   [x] Text Modelling (Filip + Tim)

    -   Bag-of-Words
        -   Unnormalized, tf-idf
    -   Word2Vec, GloVe, FastText
    -   1, 2, 3 - grams
    -   modelling.py
        -   class TextModelling (SuperClass)
            -   [x] class BOW
            -   [x] class Word2Vec
            -   [x] class GloVe
            -   [x] class FastText
            -   [x] class N-grams

-   [ ] Classifier (Sebastian)

    -   Naive Bayes, KNN, SVC
    -   sklearn.multiclass.OneVsRestClassifier
    -   https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
        -   class MultiLabelClassifier (SuperClass)
            -   [x] Naive Bayes
            -   [x] KNN
            -   [x] SVC
            -   [ ] Random Forest
            -   [ ] MLP

-   [ ] Evaluation Script (Tim)

    -   General Metrics
    -   Feature Importance
    -   Wordcloud based on Feature Importance
    -   Sentences with colored words based on Feature Importance
    -   Qualitative Results (high/low confidence/probability of estimation)
    

-   [ ] Build Pipeline (Tim)

    -   Text Modelling
    -   Classifier
    -   Evaluation

-   [ ] Deep Learning Approach (Lecture on 22.11) (Daniel)
    -   [ ] Simple MLP on TextModelling

-   [ ] Remove stop variable from start split load (just for testing so we dont need to split the whole data)


what I got from the meeting
- that we are not bound to any dataformat
- There should be a way to read conllu efficiently without reading all the information into memory that we dont need 
- it is not elegant to switch between dataformats only because one contains information we do not need 
- Analysis of system is important, not only to get one running
- Experiments with baseline, just printing e.g. f-score useless
- QUALITATIVE ANALYSIS!!
- "Looking at data and thinking about it task"
- Where are errors, with what does the model struggle -> What to do about that -> Custom Solutions
- 2 paragraphs of analysis, (Mardown file or whatever)