# pruned description lengths. 

Hypothesis: too short description may not be able to carry enough information to actually be a valid basis for classifiaction

the opposite is assumed for longer descriptions, as the probability of occurance of different "keywords" may increase, that may be not beneficial for the specific genre and may lead to a too generalized description. 

e.g. The life of Queen Victoria. [Biography]
Two love triangles intersect in ancient Pompei. [Adventure, Drama]

An epic Italian film, "Quo Vadis" influenced many of the later movies. [Drama, History]


also, some descrioptions still have "noise" e.g.

"A wealthy young man, through losing the bulk of his fortune, goes to Canada, where he encounters and loves the daughter of the scoundrel lawyer who has robbed him." (Synopsis from BIOSCOPE, November, 1930.

where you hacve author information at the end



# filter lemmatized
Additional: after lemmatization, there are certain character in the set that may lead to confusion of the model, e.g. _, ", " ", ;
also when splitting, there may be double whitespaces leading to empty strings as list items.


# additional thoughts. 
Different types of description: 

* Contentwise, e.g. actually being a summary of the movie
  * Anny works in a cigar-shop. Wholesaler Willmann fancy Anny and hire her as his housemaid. Soon the two have an intimate relationship. But Anny finds someone new. At a carnival. Her next step is to get Willman's son to get to his father's safe. Together they steal the family's fortune and have a great time doing away with all of it. Anny sink deeper and deeper in misery.
* "metadescription" e.g. 
  * An epic Italian film, "Quo Vadis" influenced many of the later movies.
  * The life of Queen Victoria.
  * 