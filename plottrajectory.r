require(tseriesChaos)
require(rgl)
walk <- NULL; run <- NULL; stand <- NULL; bike <- NULL; mixed <- NULL; data <- NULL;

d <- 3; m <- 12;

data$run <- read.csv('data/run-2009-03-02.csv',sep='\t')$accel_mag
data$walk <- read.csv('data/walk-2009-02-27-1.csv',sep='\t')$accel_mag
data$stand <- read.csv('data/standing-2009-02-27.csv',sep='\t')$accel_mag
data$bike <- read.csv('data/bike-2009-03-18.csv',sep='\t')$accel_mag
data$mixed <- read.csv('data/run-2009-03-02.csv',sep='\t')$accel_mag
data$trajectory <- read.csv('/tmp/trajectory.csv',sep='\t')

run$data <- data$run[900:2600]
walk$data <- data$walk[500:2200]
bike$data <- data$bike[6341:8110]
stand$data <- data$stand[200:700]
mixed$data <- c(data$mixed[2941:5715], data$bike[8151:10050])

walk$e <- embedd(walk$data, d=d, m=m)
run$e <- embedd(run$data, d=d, m=m)
bike$e <- embedd(bike$data, d=d, m=m)
stand$e <- embedd(stand$data, d=d, m=m)
mixed$e <- embedd(mixed$data, d=d, m=m)

p <- princomp(walk$e,scale=FALSE);

rgl.open()
rgl.clear()
rgl.linestrips(p$scores[,1:3],color=c("white"), alpha=0.2)
rgl.linestrips(data$trajectory[,1:3],col="red")
