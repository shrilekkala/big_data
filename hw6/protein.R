### *** European Protein Consumption, in grams/person-day *** ###

# This data set shows nutrition data for different types of protein in European countries around 1970.


# We want to understand in which way nutrition is different in different countries


food <- read.csv("protein.csv", row.names=1) # 1st column is country name

## scale the data

xfood <- scale(food) 

## first, consider just Red and White meat clusters


## We take a subset of the columns with p=2 variables: Red and White meat

X_subset<-food[,(1:2)]


plot(X_subset,pch=19,col=4,cex=0.2)

text(X_subset,labels=rownames(food))


# Now, let's apply the K-means clustering with 3 clusters and 10 repeats (random initializations)

grpMeat<- kmeans(X_subset, center=3, nstart=10)

plot(X_subset,pch=19,col=4,cex=0.2)

text(X_subset,labels=rownames(food),
	 col=grpMeat$cluster+1)


means<-grpMeat$centers

points(means,col=c(2,3,4),cex=2,pch=4)



## Between and Within sums of squares

# Within Sums of Squares (first cluster)

group1<-X_subset[grpMeat$cluster==1,]

sum((t(group1)-means[1,])^2)

# this value is the same as

grpMeat$withinss[1]


for(i in (1:nrow(food))){
	
	if (grpMeat$cluster[i]==1){
		
		lines(
		c(X_subset[i,1],means[1,1]),
		c(X_subset[i,2],means[1,2]),col=2)
	}
	
	
	if (grpMeat$cluster[i]==2){
		
		lines(
		c(X_subset[i,1],means[2,1]),
		c(X_subset[i,2],means[2,2]),col=3)
	}
	
	
	if (grpMeat$cluster[i]==3){
		
		lines(
		c(X_subset[i,1],means[3,1]),
		c(X_subset[i,2],means[3,2]),col=4)
	}
	
	
	
}

# Between Sums of Squares

# Overall mean

mean_overall<-apply(X_subset,2,mean)

means_between<-(t(means)-mean_overall)^2

sum(t(means_between)*grpMeat$size)

# this value is the same as

grpMeat$betweenss[1]

# Plot of cluster means


plot(X_subset,pch=19,col=grpMeat$cluster+1,cex=0.2)


means<-grpMeat$centers

points(means,col=c(2,3,4),cex=2,pch=4)

points(mean_overall[1],mean_overall[2],pch=4,lwd=3)

lines(
		c(mean_overall[1],means[3,1]),
		c(mean_overall[2],means[3,2]),
		   col=4)

lines(
		c(mean_overall[1],means[2,1]),
		c(mean_overall[2],means[2,2]),
		   col=3)

lines(
		c(mean_overall[1],means[1,1]),
		c(mean_overall[2],means[1,2]),
		   col=2)

text(c(10,12,9),c(10,8,6),grpMeat$size)

#  Total sum of squares

grpMeat$tot.withinss+grpMeat$betweenss

# which is the same as

grpMeat$totss

# which is the same as

sum((t(X_subset)-mean_overall)^2)


# Between SS == Regression sum of squares RSS
# Within SS == Sum of squared errors SSE

# Total SS == Within SS + Between SS


# Proportion of explained variance,
# Like R2 in regression, here we have

grpMeat$betweenss/(grpMeat$tot.withinss+grpMeat$betweenss)



# Let us try 7 clusters using the entire data, not only 2 columns


grp <- kmeans(xfood, centers=7, nstart=50) 

grp


plot(X_subset,pch=19,col=4,cex=0.2)

text(X_subset,labels=rownames(food),
	 col=grp$cluster+1)

# Wow, clustering makes a lot of sense here!



# Let's plot fish and read meat

plot(xfood[,"RedMeat"], xfood[,"Fish"], xlab="Red Meat", ylab="Fish",type="n")

text(xfood[,"RedMeat"], xfood[,"Fish"], labels=rownames(food), 
    col=grp$cluster+1)
    



# Hierarchical Clustering

distance<- dist(xfood,method="euclidean")

hc <- hclust(distance, method = "average")

# method can be
# complete, single, average or centroid


plot(hc)

memb <- cutree(hc, k = 7)

par(mfrow=c(1,2))

plot(X_subset,pch=19,col=4,cex=0.2)

text(X_subset,labels=rownames(food),
	 col=memb+1)

plot(X_subset,pch=19,col=4,cex=0.2)

text(X_subset,labels=rownames(food),
	 col=grp$cluster+1)

table(memb,grp$cluster)





### WEEK 8 FACTOR STUFF

food <- read.csv("protein.csv", row.names=1)

pcfood <- prcomp(food, scale=TRUE)

zfood <- predict(pcfood) 

## predict is just doing the same thing as the below:

z <- scale(food)%*%pcfood$rotation

all(z==zfood)

## implies rotations are on scale of standard deviations if scale=TRUE
## looks like PC1 is an 'average diet', PC2 is iberian

t( round(pcfood$rotation[,1:2],2) )

## do some k-means, for comparison

grpProtein <- kmeans(scale(food), centers=7, nstart=20)

## how do the PCs look?

par(mfrow=c(1,2))

plot(zfood[,1:2], type="n", xlim=c(-4,5))

text(x=zfood[,1], y=zfood[,2], labels=rownames(food), col=rainbow(7)[grpProtein$cluster])

plot(zfood[,3:4], type="n", xlim=c(-3,3))

text(x=zfood[,3], y=zfood[,4], labels=rownames(food), col=rainbow(7)[grpProtein$cluster])

## how many do we need? tough to tell

plot(pcfood, main="")

mtext(side=1, "European Protein Principle Components",  line=1, font=2)

## summary puts these scree plots on a more intuitive scale: 
	## proportion of variation explained.
summary(pcfood)
