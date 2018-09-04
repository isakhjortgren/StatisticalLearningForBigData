# StatisticalLearningForBigData
A repo that contains (IMO) the most interesting code from the course. 

The jupyter notebook is investigating an alternative usage of a generative adverserial
network as a classifier in the context of semi supervised learning. The result were 
compared to a regular CNN classifier using only the labeled data.

DetectMislabeling.R and MislabelingInvestigation.R implements a function for finding
mislabeled data and how the overall classification performance is reduced with 
mislabeled data.

MajorityVote.R implements a simple bagging algorithm that was used in the course. 

The other 3 .R scripts investigates a imbalanced data set and if different classification 
methods and sampling methods could improve the classification performance.

