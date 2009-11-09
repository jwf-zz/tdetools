require(tseriesChaos)
require(rgl)
amin <- NULL; cosmin <- NULL; maja <- NULL; rob <- NULL; jordan <- NULL; sophia <- NULL;

d <- 3; m <- 12;

amin$e <- read.csv('data/walks/amin.dat',sep=' ',colClasses="numeric",header=FALSE)
cosmin$e <- read.csv('data/walks/cosmin.dat',sep=' ',colClasses="numeric",header=FALSE)
maja$e <- read.csv('data/walks/maja.dat',sep=' ',colClasses="numeric",header=FALSE)
jordan$e <- read.csv('data/walks/jordan.dat',sep=' ',colClasses="numeric",header=FALSE)
rob$e <- read.csv('data/walks/rob.dat',sep=' ',colClasses="numeric",header=FALSE)
sophia$e <- read.csv('data/walks/sophia.dat',sep=' ',colClasses="numeric",header=FALSE)

amin$data <- t(amin$e[,1])
cosmin$data <- t(cosmin$e[,1])
maja$data <- t(maja$e[,1])
jordan$data <- t(jordan$e[,1])
rob$data <- t(rob$e[,1])
sophia$data <- t(sophia$e[,1])

y <- 2:4;
x <- 1:4000;

rgl.open()
rgl.clear()
#rgl.linestrips(amin$e[x,y],color=c("white"), alpha=0.5)
#rgl.linestrips(cosmin$e[x,y],color=c("red"), alpha=0.5)
rgl.linestrips(maja$e[x,y],color=c("blue"), alpha=0.5)
#rgl.linestrips(jordan$e[x,y],color=c("green"), alpha=0.5)
#rgl.linestrips(rob$e[x,y],color=c("yellow"), alpha=0.5)
rgl.linestrips(sophia$e[x,y],color=c("cyan"), alpha=0.5)
