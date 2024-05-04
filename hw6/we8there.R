## we8there data

library(textir) # to get the data

library(maptpx) # for the topics function

data(we8there)


# Description 
# Counts for 2804 bigrams in 6175 restaurant reviews from the site www.we8there.com. 
# Details 
# The short user-submitted reviews are accompanied by a five-star rating on four specific aspects of restaurant quality - food, service, value, and atmosphere - as well as the overall experience. The reviews originally appear in Maua and Cozman (2009). 
# Two data files: we8thereCounts and we8thereRatings

dim(we8thereCounts)

# 6266 reviews
# 2640 bigrams (pair of words)


dim(we8thereRatings)

names(we8thereRatings)

# ratings that relate to food, service, value, atmosphere, overall experience


# Can we try K-means?  K-means assume that the features are continuous, here we have a sparse matrix of counts 

# We can try produce something that K-means can work with

# Let's first normalize by the length of the review (number of bigrams) and then standardize

fs <- scale(as.matrix( we8thereCounts/rowSums(we8thereCounts) ))

# takes time, because you're making it dense from sparse


#  upon what scale to measure `words' for k-means is unclear (the answer is  actually: don't use kmeans; use a multinomial mixture instead).


# Let's fit K-means with 4 clusters

kmfs <- kmeans(fs,4)  

## both this and topic modelling takes a long time...
## you're fitting massively high dimensional models (K*ncol(x))
## there are approximate distributed algorithms out there
## `stochastic gradient descent', etc...
## for really big data, I think you just focus on subsamples, most common words, etc.
## all these methods find are the dominant sources of variation, 
## so those should be present in small subsamples

## interpretation: we can see the words with cluster centers
## highest above zero (these are in units of standard deviation of f)



print(apply(kmfs$centers,1,function(c) colnames(fs)[order(-c)[1:20]]))


#  K-means  gives a sensible answer but it is far from an ideal model for our data (our data are counts!)



## topic modelling.  Treat counts as actual counts!
## i.e., model them with a multinomial
## we'll use the topics function in maptpx (there are other options out there)

## you need to convert from a Matrix to a `slam' simple_triplet_matrix
## luckily, this is easy.

x <- as.simple_triplet_matrix(we8thereCounts)

# to fit, just give it the counts, number of `topics' K, and any other args

tpc <- topics(x,K=10) 


dim(tpc$theta)
colSums(tpc$theta)

dim(tpc$omega)
rowSums(tpc$omega)

## choosing the number of topics
## If you supply a vector of topic sizes, it uses a Bayes factor to choose
## (BF is like exp(-BIC), so you choose the bigggest BF)
## the algo stops if BF drops twice in a row

tpcs <- topics(x,K=5*(1:5), verb=10) # it chooses 10 topics 

## interpretation
# summary prints the top `n' words for each topic,
# under ordering by `topic over aggregate' lift:
#    the topic word prob over marginal word prob.

summary(tpcs, n=10) 

# this will promote rare words that with high in-topic prob

# alternatively, you can look at words ordered by simple in-topic prob
## the topic-term probability matrix is called 'theta', 
## and each column is a topic
## we can use these to rank terms by probability within topics

rownames(tpcs$theta)[order(tpcs$theta[,1], decreasing=TRUE)[1:10]]

rownames(tpcs$theta)[order(tpcs$theta[,2], decreasing=TRUE)[1:10]]



#etc

## Wordles!
## use the wordcloud library to plot a few

library(wordcloud)

## we'll size the word proportional to its in-topic probability
## and only show those with > 0.004 omega
## (it will still likely warn that it couldn't fit everything)

par(mfrow=c(1,2))

wordcloud(row.names(tpcs$theta), 
	freq=tpcs$theta[,1], min.freq=0.004, col="maroon")

wordcloud(row.names(tpcs$theta), 
	freq=tpcs$theta[,2], min.freq=0.004, col="navy")

## interpret the relationship between topics and overall rating

library(gamlr)

## omega is the n x K matrix of document topic weights
## i.e., how much of each doc is from each topic
## we'll regress overall rating onto it

stars <- we8thereRatings[,"Overall"]

tpcreg <- gamlr(tpcs$omega, stars)

# number of stars up or down for moving up 10\% weight in that topic

coef(tpcreg)*0.1

regtopics.cv <- cv.gamlr(tpcs$omega, stars,lambda.min.ratio=10^{-4})

## give it the word %s as inputs

x <- 100*we8thereCounts/rowSums(we8thereCounts)

regwords.cv <- cv.gamlr(x, stars)

par(mfrow=c(1,2))

plot(regtopics.cv)

mtext("topic regression", font=2, line=2)

plot(regwords.cv)

mtext("bigram regression", font=2, line=2)

# min OOS MSE

min(regtopics.cv$cvm)

min(regwords.cv$cvm)



