# NLP Toxic Comment Classification
## Problem Statement : 
Given a comment from social media platform, our task is to identify if it has any toxic content and classify it to belong to one or more of the following categories.
- Toxic
- Angry
- Sad
- Fear
- Joy
- Shame
- Disgust
- Shame
- Neutral 


## Our Approach :
We have devised a prototype that is capable of classifying a comment/text on the basis of above mentioned categories. We have made our own Dataset for this multilabel classification problem by merging multiple pre-existing datasets into one dataset. The dataset has 8 labels that represent the above comment categories, but the project is going to focus on a seventh label that represents the general toxicity of the comments.The project is done with Python and Jupyter notebooks, which are all attached.
- Explore the dataset to get a better picture of how the labels are distributed, how they correlate with each other, and what defines toxic or clean comments or comments from other categories. 
- Create a baseline score with a simple logistic regression classifier. 
- Explore the effectiveness of multiple machine learning approaches and select the best for this problem.
- Using Python web GUI - Streamlit deploy the model.

Disclaimer: The dataset has extremely offensive language that will show up during exploratory data analysis done in the Jupyter notebook itself.


## Algorithms and Techniques :

To perform the classification of toxic comment in our application, we have utilized various algorithms used for Natural Language Processing, which are stated below:

- Logistic Regresssion
- Multinomial Naive Bayes
- Support Vector Machine
- Minimum Edit Distance




## Application :
- Home page

![s3](https://user-images.githubusercontent.com/51513456/166161152-2fd26085-56f9-46ea-a2b0-e3882502d30e.jpeg)

- Comment Type "Joy"

![s4](https://user-images.githubusercontent.com/51513456/166161159-96a4cee6-edb2-485c-b054-12b7d7e8a3c7.jpeg)

- Comment Type "Sadness"

![s2](https://user-images.githubusercontent.com/51513456/166161167-0550dfd6-ddb7-4823-9c63-b7b7362ee47a.jpeg)

- Comment Type "Toxic" with Suggested Correction

![s1](https://user-images.githubusercontent.com/51513456/166161171-019073ed-bee4-4e62-9731-66dfa0080ad7.jpeg)

