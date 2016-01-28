# Apache-Spark-Movie-Recommendation-System

This particular recommendation system uses IMDB dataset available online. The dataset comprises of 10M entries. You can download the same from <a href='https://drive.google.com/folderview?id=0B7hi_XaI-t-9WF9pMUJoM3VfMDA&usp=sharing'>here</a>. OR you can google for other available datasets from IMDB. Remember to keep both data files in the save directory as script.

The application is developed in Python on Apache Spark.

Recommendation works on the principle of collaborative filtering.More about collaborative filtering can be learn <a href = 'http://recommender-systems.org/collaborative-filtering/'>here</a> The final result is obtained in the form of RDD which can be saved to a text file with instruction rdd.saveAsTextFile('name.txt'). 
