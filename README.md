# DisRFKQR
An R package for "Communication-Efficient Nonparametric Quantile Regression via Random Features".

## Example 
library(DisRFKQR)
library(MASS)

################################## This is the R code of example 1 in the main text (example 2 is also similar with an different model)#########################



########## Experiment Setup ############

# Training, testing and validation sample size
N_tr<-5000
N_te<-10000
N_va<-1000
N<-N_tr+N_va+N_te

# Generating model
d=3
X<- matrix(runif(N*d),ncol=d)
f_x<-exp(-X[,1]+X[,2])-X[,2]*X[,3]+abs(apply(X,1,mean))
epsilo<- c(rnorm(N))
Y<-f_x+epsilo


# Training, validation and testing data
X_tr<-X[1:N_tr,]
f_x_tr<-f_x[1:N_tr]
Y_tr<-Y[1:N_tr]

X_va<-X[(N_tr+1):(N_tr+N_va),]
f_x_va<-f_x[(N_tr+1):(N_tr+N_va)]
Y_va<-Y[(N_tr+1):(N_tr+N_va)]

X_te<-X[(N_tr+N_va+1):N,]
f_x_te<-f_x[(N_tr+N_va+1):N]
Y_te<-Y[(N_tr+N_va+1):N]

# Random feature mapping
L<-50 # number of random features
phi_x<-RF_mapping(X,L,d)
phi_x_tr<-phi_x[1:N_tr,]
phi_x_va<-phi_x[(N_tr+1):(N_tr+N_va),]
phi_x_te<-phi_x[(N_tr+N_va+1):N,]

# Parameter setting
tau<-0.5 # quantile level
n<-100 # local sample size   
M<-N_tr/n # number of local machines
iter<-10 # algorithm iteration 
admm_iter<-1000 # admm iteration
gamma<-0.001 
r_0<-rep(0,n)
u_0<-rep(0,n)
########## Model training ##########

# Tuning parameters on validate sample
lambda_list<-10^seq(-10,1,0.5)

PQE_list<-c()
for (lambda in lambda_list){
  theta<-DKQR_RF(phi_x_tr,Y_tr,r_0,u_0,lambda,tau,M,gamma,L,admm_iter,iter) 
  PQE<-mean(qr_loss(Y_va-as.matrix(phi_x_va%*%theta[,iter]))) # calculate the PQE 
  PQE_list<-c(PQE_list,PQE)
  #print(PQE)
}

lambda_best<-lambda_list[which.min(PQE_list)] # select the best lambda which minimize the PQE 

# Fit the model with the best lambda
theta_best<-DKQR_RF(phi_x_tr,Y_tr,r_0,u_0,lambda_best,tau,M,gamma,L,admm_iter,iter)

# Evaluation on the testing data
PQE_te<-mean(qr_loss(Y_te-as.matrix(phi_x_te%*%theta_best[,iter]))) # Estimated PQE on the testing data
PQE_te_true<-mean(array(unlist(lapply((Y_te-as.double(f_x_te)-qnorm(tau)),qr_loss)))) # True PQE on the testing data

print(c(PQE_te,PQE_te_true))


