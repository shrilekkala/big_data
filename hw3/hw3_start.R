# ***** AMAZON REVIEWS 

# READ REVIEWS

data<-read.table("Review_subset.csv",header=TRUE)
dim(data)

# 13319 reviews
# ProductID: Amazon ASIN product code
# UserID:  id of the reviewer
# Score: numeric from 1 to 5
# Time: date of the review
# Summary: text review
# nrev: number of reviews by this user
# Length: length of the review (number of words)

# READ WORDS

words<-read.table("words.csv")
words<-words[,1]
length(words)
#1125 unique words

# READ text-word pairings file

doc_word<-read.table("word_freq.csv")
names(doc_word)<-c("Review ID","Word ID","Times Word" )
# Review ID: row of the file  Review_subset
# Word ID: index of the word
# Times Word: number of times this word occurred in the text


#################################################
# Let's define the binary outcome

# Y=1 if the rating was 5 stars

# Y=0 otherwise

Y<-as.numeric(data$Score==5)

# (a) Use only product category as a predictor

library(gamlr)

source("naref.R") 

# Cast the product category as a factor
data$Prod_Category<-as.factor(data$Prod_Category)

class(data$Prod_Category)

# look inside naref.R; it applies to every factor variable:
#   > factor(x,levels=c(NA,levels(x)),exclude=NULL) 
# Since product category is a factor, we want to relevel it for the LASSO. We want each coefficient to be an intercept for each factor level rather than a contrast.

levels(data$Prod_Category)

data$Prod_Category<-naref(data$Prod_Category)

levels(data$Prod_Category)

# Create a design matrix using only products

products<-data.frame(data$Prod_Category)


x_cat<-sparse.model.matrix(~., data=products)[,-1]

# Sparse matrix, storing 0's as .'s 
# We removed intercept so that each category is standalone, not a contrast relative to the baseline category


colnames(x_cat)<-levels(data$Prod_Category)[-1]
# let's call the columns of the sparse design matrix as the product categories


# Let's fit the LASSO with just the product categories

lasso1<- gamlr(x_cat, 	y=Y,standardize=FALSE,family="binomial",
lambda.min.ratio=1e-3)

# (2) Fit a LASSO with all 142 product categories and 1125 words 


library(gamlr)

spm<-sparseMatrix(i=doc_word[,1],j=doc_word[,2],x=doc_word[,3],dimnames=list(id=1:nrow(data),words=words))

dim(spm)

# 13319 reviews using 1125 words


x_cat2<-cbind(x_cat,spm)

lasso2 <- gamlr(x_cat2,
				   y=Y,
				   lambda.min.ratio=1e-3,
				   family="binomial",
				   verb=TRUE)


# (3) cross-validation

cv.fit <- cv.gamlr(x_cat2,
				   y=Y,
				   lambda.min.ratio=1e-3,
				   family="binomial",
				   verb=TRUE)


