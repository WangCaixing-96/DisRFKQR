#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

int max(int u, int v){
  int s;
  if(u<=v)
    s=v;
  else
    s=u;
  return s;
}


arma::vec shrink(arma::vec u, arma::vec v){
  arma::vec w=(1+sign(u-v))/2%(u-v);
  return w;
}

arma::vec Itau(arma::vec u, arma::vec v, double tau){
  int n=u.n_elem;
  arma::vec I=arma::zeros(n);
  for(int i=0;i<n;i++){
    if(u(i)<v(i)) I(i)=1-tau;
    else I(i)=-tau;
  }
  return I;
}


arma::vec grad_update(arma::mat phi_x, arma::vec Y, arma::vec theta, double lambda, double tau){
  int n=Y.n_elem;
  arma::vec hat_y=phi_x*theta;
  arma::vec grad=phi_x.t()*Itau(Y, hat_y, tau)/n+2*lambda*theta;
  return grad;
}

//[[Rcpp::export]]
arma::vec dis_grad(arma::mat phi_x, arma::vec Y, arma::vec theta, double lambda, double tau, int S, int M){
  int N=Y.n_elem;
  int n=N/M;
  arma::mat grad_tot=arma::zeros(S,M);

  for(int i=0;i<M;i++){
    arma::mat phi_xi=phi_x.rows(n*i,n*i+n-1);
    arma::vec Yi=Y.subvec(n*i,n*i+n-1);
    arma::vec gradi=grad_update(phi_xi,Yi,theta,lambda,tau);
    grad_tot.col(i)=gradi;
  }
  arma::vec grad_mean=mean(grad_tot,1);

  arma::mat phi_x1=phi_x.rows(0,n-1);
  arma::vec Y1=Y.subvec(0,n-1);
  arma::vec grad_1=grad_update(phi_x1,Y1,theta,lambda,tau);

  arma::vec grad_dif=grad_mean-grad_1;
  return grad_dif;

}

//[[Rcpp::export]]
arma::mat ADMM(arma::mat phi_x, arma::vec Y, arma::vec grad_dif, arma::vec r_0, arma::vec u_0, double lambda, double tau, double beta, int S, int max_iter){
  int n=Y.n_elem;
  arma::vec r_old=r_0;
  arma::vec u_old=u_0;
  arma::vec theta_old=arma::zeros(S);
  int L=max(S,n);
  arma::mat adm_para=arma::zeros(L,3);
  int iter=0;
  while(iter<max_iter){
    arma::vec theta=((phi_x.t()*phi_x+2*(lambda/beta)*arma::eye(S,S)).i())*(phi_x.t()*(Y-r_old-u_old/beta)-grad_dif/beta);
    arma::vec r=shrink(Y-phi_x*theta-u_old/beta,tau/(n*beta)*arma::ones(n))-shrink(-Y+phi_x*theta+(u_old/beta),(1-tau)/(n*beta)*arma::ones(n));
    arma::vec u=u_old+beta*(phi_x*theta+r-Y);
    r_old=r;
    u_old=u;
    theta_old=theta;
    iter=iter+1;
  }
  adm_para.col(0).subvec(0,S-1)=theta_old;
  adm_para.col(1).subvec(0,n-1)=r_old;
  adm_para.col(2).subvec(0,n-1)=u_old;

  return adm_para;
}

//[[Rcpp::export]]
arma::mat DKQR_RF(arma::mat phi_x, arma::vec Y, arma::vec r_0, arma::vec u_0, double lambda, double tau, int M, double beta, int S, int max_iter_admm, int max_iter_dis){
  int N=Y.n_elem;
  int n=N/M;
  arma::vec theta_old=arma::zeros(S);
  arma::vec r_old=arma::zeros(n);
  arma::vec u_old=arma::zeros(n);

  arma::vec grad_dif_old=arma::zeros(S);
  arma::mat phi_x1=phi_x.rows(0,n-1);
  arma::vec Y1=Y.subvec(0,n-1);
  arma::mat para_old=ADMM(phi_x1,Y1,grad_dif_old,r_0,u_0,lambda,tau,beta,S,max_iter_admm);
  theta_old=para_old.col(0).subvec(0,S-1);
  r_old=para_old.col(1).subvec(0,n-1);
  u_old=para_old.col(2).subvec(0,n-1);
  arma::mat tmp=arma::zeros(S,max_iter_dis);
  //tmp.col(0)=0.5*arma::ones(S);
  tmp.col(0)=theta_old;

  for(int i=1;i<max_iter_dis;i++){
    arma::vec grad_dif_i=dis_grad(phi_x,Y,theta_old,lambda,tau,S,M);
    arma::mat para=ADMM(phi_x1,Y1,grad_dif_i,r_0,u_0,lambda,tau,beta,S,max_iter_admm);
    theta_old=para.col(0).subvec(0,S-1);
    tmp.col(i)=theta_old;
  }

  return(tmp);

}



//[[Rcpp::export]]
arma::vec dis_grad_pilot(arma::mat phi_x, arma::vec Y, arma::mat phi_x_p, arma::vec Y_p, arma::vec theta, double lambda, double tau, int S, int M){
  int N=Y.n_elem;
  int n=N/M;
  arma::mat grad_tot=arma::zeros(S,M);

  for(int i=0;i<M;i++){
    arma::mat phi_xi=phi_x.rows(n*i,n*i+n-1);
    arma::vec Yi=Y.subvec(n*i,n*i+n-1);
    arma::vec gradi=grad_update(phi_xi,Yi,theta,lambda,tau);
    grad_tot.col(i)=gradi;
  }
  arma::vec grad_mean=mean(grad_tot,1);

  arma::vec grad_p=grad_update(phi_x_p,Y_p,theta,lambda,tau);

  arma::vec grad_dif_p=grad_mean-grad_p;

  return grad_dif_p;

}



//[[Rcpp::export]]
arma::mat DKQR_RF_Pilot(arma::mat phi_x, arma::vec Y, arma::mat phi_x_p, arma::vec Y_p, arma::vec r_0, arma::vec u_0, double lambda, double tau, int M, double beta, int S, int max_iter_admm, int max_iter_dis){
  int N=Y.n_elem;
  int n=N/M;
  arma::vec theta_old=arma::zeros(S);
  arma::vec r_old=arma::zeros(n);
  arma::vec u_old=arma::zeros(n);

  arma::vec grad_dif_old=arma::zeros(S);
  arma::mat para_old=ADMM(phi_x_p,Y_p,grad_dif_old,r_0,u_0,lambda,tau,beta,S,max_iter_admm);
  theta_old=para_old.col(0).subvec(0,S-1);
  r_old=para_old.col(1).subvec(0,n-1);
  u_old=para_old.col(2).subvec(0,n-1);
  arma::mat tmp=arma::zeros(S,max_iter_dis);
  //tmp.col(0)=arma::zeros(S);
  tmp.col(0)=theta_old;

  for(int i=1;i<max_iter_dis;i++){
    arma::vec grad_dif_i=dis_grad_pilot(phi_x,Y,phi_x_p,Y_p,theta_old,lambda,tau,S,M);
    arma::mat para=ADMM(phi_x_p,Y_p,grad_dif_i,r_0,u_0,lambda,tau,beta,S,max_iter_admm);
    theta_old=para.col(0).subvec(0,S-1);
    tmp.col(i)=theta_old;
  }

  return(tmp);

}
