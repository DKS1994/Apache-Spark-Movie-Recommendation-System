
# coding: utf-8

# In[5]:

import sys
import os


# In[6]:

#It is assumed that you have folder named 'data' which has another folder within it with name 'COMPTON' which contains ratings and movies files


# In[7]:

baseDir = os.path.join('data')
inputPath = os.path.join('COMPTON')


# In[8]:

#Ratings file conatins data in the form=> UserID::MovieID::Rating::TimeStamp


# In[9]:

ratingsFileName=os.path.join(baseDir,inputPath,'mooc-ratings.dat')


# In[10]:

#Movies file contains data in the form=> MovieID::Title::Genres


# In[11]:

moviesFileName=os.path.join(baseDir,inputPath,'mooc-movies.dat')


# In[12]:

#Building RDD from given files


# In[13]:

numPartitions = 2
rawRatings = sc.textFile(ratingsFileName).repartition(2)
rawMovies = sc.textFile(moviesFileName)


# In[14]:

#get_ratings_tuple returns ratingsRDD in the form=>UserID::MovieID::Rating


# In[15]:

def get_ratings_tuple(entry):
    items = entry.split('::')
    return int(items[0]),int(items[1]),float(items[2])
ratingsRDD = rawRatings.map(get_ratings_tuple).cache()


# In[16]:

#get_movies_tuple returns moviesRDD in the form=>MovieID::Title


# In[17]:

def get_movies_tuple(entry):
    items = entry.split('::')
    return int(items[0]),items[1]
moviesRDD = rawMovies.map(get_movies_tuple).cache()


# In[18]:

#get_average_rating returns (MovieID,(number of ratings,average of rating tuple))


# In[19]:

def get_average_rating(IDandRatingsTuple):
    countOfRatings = len(IDandRatingsTuple[1])
    averageOfRatings = float(sum(IDandRatingsTuple[1]))/countOfRatings
    return (IDandRatingsTuple[0],(countOfRatings,averageOfRatings))


# In[20]:

#movieIDsWithRatingsRDD consists of movieID and the tuple of all ratings assigned to it
# movieIDsWithRatingsRDD has the form => (MovieID,(Rating1,Rating2,.....))


# In[21]:

movieIDsWithRatingsRDD = ratingsRDD.map(lambda x : (x[1],x[2])).groupByKey()


# In[22]:

IDRatingsCountAndAverage = movieIDsWithRatingsRDD.map(get_average_rating)


# In[26]:

#movieNameWithAvgRatingsRDD has the form=>(average rating,Title,number of ratings)


# In[29]:

movieNameWithAvgRatingsRDD = (moviesRDD
                             .join(IDRatingsCountAndAverage).map(lambda x:(x[1][1][1],x[1][0],x[1][1][0])))


# In[31]:

#splitting ratingsRDD in 3 RDDs.


# In[32]:

trainingRDD , validationRDD , testRDD = ratingsRDD.randomSplit([6,2,2],seed = 0L)


# In[33]:

import math


# In[35]:

#compute_error computes Root Mean Square Error


# In[36]:

def compute_error(predictedRDD,actualRDD):
    predictedRatings = predictedRDD.map(lambda (x,y,z):((x,y),z))
    actualRatings = actualRDD.map(lambda (x,y,z):((x,y),z))
    
    combinedRDD = (predictedRatings.join(actualRatings)).map(lambda (x,y):(y[0]-y[1])**2)
    count = combinedRDD.count()
    summation = combinedRDD.sum()
            
    return math.sqrt(float(summation)/count)
    


# In[37]:

from pyspark.mllib.recommendation import ALS


# In[39]:

#validationForPredictedRDD has the form=>(UserID,MovieID)


# In[40]:

validationForPredictedRDD = validationRDD.map(lambda (x,y,z):(x,y))


# In[41]:

seed =5L
iterations = 5
regularizationParameter = 0.1
ranks = [4,8,12]
errors = [0,0,0]
err = 0
tolerance = 0.02

minError = float('inf')
bestRank =-1
bestIteration = -1


# In[42]:

#calculating bestRank for our training model out of given ranks [4,8,12]


# In[43]:

for rank in ranks:
    model = ALS.train(trainingRDD,rank,seed=seed,iterations=iterations,lambda_=regularizationParameter)
    predicted_ratings = model.predictAll(validationForPredictedRDD)
    
    error = compute_error(predicted_ratings,validationRDD)
    errors[err]=error
    err += 1
    if error < minError:
        minError = error
        bestRank = rank
        
        


# In[ ]:

#building best training model from bestRank obtained


# In[49]:

bestModel = ALS.train(trainingRDD,bestRank,seed=seed,iterations=iterations,lambda_=regularizationParameter)


# In[50]:

#building testForPredictingRDD for test error in prediction


# In[51]:

testForPredictingRDD = testRDD.map(lambda (x,y,z):(x,y))


# In[53]:

#predicting ratings for testForPredictingRDD


# In[54]:

PredictingTestRDD = bestModel.predictAll(testForPredictingRDD)


# In[56]:

#Testing ERROR in the obtaied ratings and original RDD


# In[57]:

testError = compute_error(PredictingTestRDD,testRDD)


# In[58]:

print testError


# In[59]:

#For comparing our testError against the average rating for all movies in testRDD


# In[60]:

testCount = testRDD.count()
testRDDratingsAvg = (float(testRDD.map(lambda (x,y,z):z).sum())/testCount)


# In[61]:

testAvgRDD = testRDD.map(lambda (x,y,z):(x,y,testRDDratingsAvg))


# In[62]:

testERR = compute_error(testAvgRDD,testRDD)


# In[64]:

print testERR


# In[65]:

#Now predicting movies for a new user created by ourself with userID myUserID equal to 0


# In[93]:

myUserID = 0


# In[94]:

#Giving rating to different movies by making RDD with the form same as to trainingRDD. 
#You can rate movies by adding tuple (userID,movieID,rating)
#You can look for movies ID from moviesRDD or directly from movies.dat file.I leave this to you to figure out.


# In[115]:

myRatedMovies = [
     (myUserID,993,4),(myUserID,983,4.5),(myUserID,789,4),(myUserID,539,3),(myUserID,1438,5),(myUserID,1195,5),(myUserID,1088,4),(myUserID,651,3),(myUserID,551,2),(myUserID,662,5)
     # The format of each line is (myUserID, movie ID, your rating)
     # For example, to give the movie "Star Wars: Episode IV - A New Hope (1977)" a five rating, you would add the following line:
     #   (myUserID, 260, 5),
    ]


# In[116]:

myRatedRDD = sc.parallelize(myRatedMovies)


# In[117]:

#The number of movies in our training set


# In[118]:

print trainingRDD.count()


# In[119]:

#Adding our movies to trainingRDD


# In[120]:

trainingSetWithMyRatings = trainingRDD.union(myRatedRDD)


# In[121]:

#The number of movies after adding to training set


# In[122]:

print trainingSetWithMyRatings.count()


# In[123]:

#Building training model from our new training set RDD with bestRank calculated above


# In[124]:

myRatingModel = ALS.train(trainingSetWithMyRatings,bestRank,seed=seed,iterations=iterations,lambda_=regularizationParameter)


# In[125]:

#Making predictions with our model on testRDD


# In[126]:

PredictionsOnTestRDD = myRatingModel.predictAll(testForPredictingRDD)


# In[127]:

print PredictionsOnTestRDD.count()


# In[128]:

#Computing accuracy for our new model


# In[129]:

accuracyIs = compute_error(PredictionsOnTestRDD,testRDD)


# In[130]:

print accuracyIs


# In[131]:

#Predicting our ratings for unrated movies by us!
#myRatedMoviesWithoutRatings has the form=>(UserID,MovieID)


# In[132]:

myRatedMovies = sc.parallelize(myRatedMovies).map(lambda (x,y,z):(x,y))


# In[133]:

print myRatedMovies.take(3)


# In[135]:

#Obtaining movies which we didn't rated in the form=>(UserID,MovieID)


# In[136]:

myUnratedMovies = moviesRDD.map(lambda (x,y):(myUserID,x)).subtract(myRatedMovies)


# In[137]:

print moviesRDD.count()


# In[138]:

print myUnratedMovies.count()


# In[140]:

#Predicting ratings for movies we didn't rate by our model myRatingModel


# In[141]:

predictionsForUnratedMovies = myRatingModel.predictAll(myUnratedMovies)


# In[143]:

#PredictedRDD has the form=>(MovieID,predicted_Ratings)


# In[144]:

predictedRDD = predictionsForUnratedMovies.map(lambda (x,y,z):(y,z))


# In[72]:

print IDRatingsCountAndAverage.take(3)


# In[146]:

#movieCounts has the form=>(MovieID,number_of_ratings)


# In[147]:

movieCounts = IDRatingsCountAndAverage.map(lambda (x,y):(x,y[0]))


# In[148]:

print movieCounts.take(3)


# In[149]:

#movieCountsWithPredictedRDD has the form=>(MovieID,(number_of_ratings,predicted_ratings))


# In[150]:

movieCountsWithPredictedRDD = movieCounts.join(predictedRDD)


# In[151]:

#movieNameCountPredictedRatings has the form=>(MovieID,(Title,(number_of_ratings,predicted_ratings)))


# In[152]:

movieNameCountPredictedRatings = moviesRDD.join(movieCountsWithPredictedRDD)


# In[154]:

#predictedRatingsWithName has the form=>(predicted_Ratings,Title,number_of_ratings)


# In[155]:

predictedRatingsWithName = movieNameCountPredictedRatings.map(lambda x:(x[1][1][1],x[1][0],x[1][1][0]))


# In[158]:

#For better recommendation we take movies which have more than 75 ratings by other users.


# In[159]:

recommendedMoviesFromHighestRating = predictedRatingsWithName.filter(lambda x : x[2]>75)


# In[160]:

#Top ten recommended movie for ourself!!!


# In[163]:

print recommendedMoviesFromHighestRating.sortBy(lambda x:-x[0]).take(10)


# In[ ]:



