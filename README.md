#Cell Differentiation
This project analyzes RNA data with over 1 million dimensions to find structure.
In particular, the goal is to cluster 272 cells into different cell-types, and see how the cell-types are related to each other.

First we use PCA to reduce the data's dimension, and k-means to cluster the cells.
Then we determine the most important RNA markers for distinguishing between the cell-types by using multi-class logistic regression.

Finally, we plot a differentiation tree (by computing a minimum spanning tree) to study the similarity between the cell-types.

##Data
Each row of the .csv file corresponds to one of the 272 cells, and each column corresponds to the number of reads within a different section of the RNA. 
We consider over 1 million sections of RNA, and the rows are normalized to sum to 1. 
