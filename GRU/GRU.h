#ifndef GRU_H
#define GRU_H 1

#include "Arduino.h"

#define sigmoid(x) (float) 1.0/(1.0 + exp(-(float)x))
#define th(x) (float) 2.0*sigmoid(2.0*(float)x) - 1.0



class GRU {
private:
  float *h; // state
  float *u; // update gate
  float *r; // reset gate
  float *h_c; // candidate state

  float **Wu, **Uu, *bu; // parameters of update gate
  float **Wr, **Ur, *br; // parameters of reset gate
  float **Wc, **Uc, *bc; // parameters of candidate state
  
  int dim_input_;
  int dim_hidden_;
  int num_steps_;
  
  float* mulMatVec(float **M, float *x, int num_rows, int num_cols); 
  
  void calculateUpdateGate(float *x);
  void calculateResetGate(float *x);
  void calculateCandidateState(float *x);
  
  int created_; 
  long step_;
  
public:

 GRU(int dim_input, int dim_hidden, int n_steps);
 ~GRU();
 void setParameters(float *theta);
 float *getParameters();
 int getNumParameters();
 int getStateSize();
 int getInputSize();
 int getStatus();
 long getStep() { return step_; }
 void setStatus(int status);
 
 
 void setState(float *theta); 
 void resetState(); 
 void input(float *input);
 float *getState(); 

 float rand01()
    {
          return (float) random(1, 1024)/1024.0;
    }

 float randn(float mu, float sigma)
    {
      float x1 = rand01();
      float x2 = rand01();
      return mu + sqrt(-2.0 * log(x1)*sigma) * cos(2.0* PI * x2);
    }

};
#endif

