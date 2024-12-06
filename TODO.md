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

-   [x] Classifier (Sebastian)

    -   Naive Bayes, KNN, SVC
    -   sklearn.multiclass.OneVsRestClassifier
    -   https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
        -   class MultiLabelClassifier (SuperClass)
            -   [x] Naive Bayes
            -   [x] KNN
            -   [x] SVC
            -   [x] Random Forest
            -   [x] MLP

-   [ ] Evaluation Script 

    -   [x] General Metrics
    -   [x] Bar Feature Importance (Filip)
    -   [x] Wordcloud based on Feature Importance (Filip)
    -   [x] Sentences with colored words based on Feature Importance (Tim)
    -   [ ] Qualitative Results (high/low confidence/probability of estimation) (Tim)
        -   Completely wrong predictions (negative hits)
        -   predict_proba() for each class
    -   Confusion Matrix per genre (Sebastian)
    -   Correlation matrix per row accoress classifiers (Sebastian)
    -   Plot embeddings per description in 2D space (PCA, t-SNE), color based on classes (Daniel)
    -   Random Guess
        -   {'jaccard': 2.635, 
            'hamming': 0.102,
            'accuracy': 0.0,
            'f1': 4.294e-05,
            'precision': 3.904e-05, 
            'recall': 4.880e-05,
            'at_least_one': 0.0001}


-   [ ] Run Experiments (Tim)
    -   [ ] BOW / Word2Vec + 
        -   [ ] Small Decision Tree (Visualize Tree)
        -   [ ] Logistic Regression
        -   [ ] Naive Bayes
        -   [ ] Support Vector Machine


-   [ ] Deep Learning Approach (Lecture on 22.11) (Daniel, Filip)
    -   [ ] Simple MLP on TextModelling
    -   [X] Finetune pretrained LLM (DistilBERT from Huggingface) on RAW dataset

    -   [ ] Handle Division by Zero and Low Support Classes
        -   [ ] Investigate division by zero and precision = 0.00 issues in low support classes.
        -   [ ] Potential solutions:
            -   [ ] Get rid of low support classes?
            -   [ ] Adjust weights of imbalanced classes.
                -   [ ] Implement a **CustomTrainer**. Refer to Huggingface discussion: [CustomTrainer Discussion](https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067).
                -   [ ] Use `BCEWithLogitsLoss` or another suitable custom loss function.
        -   [ ]  Run the model with more data to improve class representation.

    -   [ ] Low Support Classes Explanation
        -   [ ] Document specific low support classes (e.g., "music," "musical," "western," "sport", "film-noir").
        -   [ ] Address issues related to these classes using one or more of the strategies above.

    -   [X] Threshold Tuning
        -   Prob. threshold of 0.425 mproves recall while keeping precision.
        -   Select the highest probability class if no others meet the threshold.
        -   Improves performance by a few percentage points.

-   [ ] Remove stop variable from start split load (just for testing so we dont need to split the whole data)


## Final TODO Milestone II:

- [ ] Short overview of what we tried (classifiers and modelling approaches). 
- [ ] Basic scores per modelling + classifier (confusion matrix per class, etc)
- [ ] Take best classifier(s) and describe what we found, where is the model good and where is it bad? 
  - Feature importance
  - Example of good and bad examples (text)
  - Confusion matrix and other plots

## Task for Final
- [ ] Drop Nonsense descriptions (how to detect) (maybe syntax analysis)
- [ ] Maybe drop to short/long description
- [ ] remove common shared words across classes (for bOw and stuff like this)
- [ ] ??

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