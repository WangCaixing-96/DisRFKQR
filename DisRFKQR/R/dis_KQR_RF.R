qr_loss<-function(u){
  return(u*(tau-ifelse(u<=0,1,0)))
}
square_loss<-function(u){
  return(u^2)
}

iterloss<-function(u, loss_name){
  if(loss_name=="PQE")
     tmp<-apply(u,2,qr_loss)
  else
     tmp<-apply(u,2,square_loss)
  loss<-apply(tmp,2,mean)
  return(loss)
}

RF_mapping<-function(X, L, d){
  phi_x<-c()
  for (j in 1:L){
    w<- mvrnorm(1, rep(0,d), diag(d))
    b=runif(1,0,2*pi)
    if(d>1)
    phi_xj<-sqrt(2)*unlist(lapply((X%*%w+b),cos))
    else
    phi_xj<-sqrt(2)*cos(w*X+b)
    phi_x<-append(phi_x,phi_xj)
  }

  phi_x<-matrix(as.double(phi_x),ncol=L)/sqrt(L)
  return(phi_x)
}
