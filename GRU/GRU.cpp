#include "GRU.h"



GRU::GRU(int dim_input, int dim_hidden, int num_steps) : created_(0), step_(0), num_steps_(0)
{
   dim_input_ = dim_input;
   dim_hidden_ = dim_hidden;
   
   Wu = new float *[dim_hidden];   
   Wr = new float *[dim_hidden];   
   Wc = new float *[dim_hidden];
   
   Uu = new float *[dim_hidden];   
   Ur = new float *[dim_hidden];   
   Uc = new float *[dim_hidden];
   
   for (int i = 0; i < dim_hidden; i++)
     {
       Wu[i] = new float[dim_input];
       Wr[i] = new float[dim_input];
       Wc[i] = new float[dim_input]; 
       Uu[i] = new float[dim_hidden];
       Ur[i] = new float[dim_hidden];
       Uc[i] = new float[dim_hidden];
     }  
    bu = new float[dim_hidden];
    br = new float[dim_hidden];
    bc = new float[dim_hidden];

    u = new float[dim_hidden];
    r = new float[dim_hidden];
    h_c = new float[dim_hidden];
    h = new float[dim_hidden];
    
    
    resetState();
    float *theta = new float[getNumParameters()];
    for (int i = 0; i < getNumParameters(); i++)
      theta[i] = (float) randn(0, 1.0);      
    setParameters(theta);    
    delete[] theta;    
    
    created_ = 1;      
    step_ = 0;      
    num_steps_ = num_steps;  

}

GRU::~GRU()
{
  delete[] bu;
  delete[] br;
  delete[] bc;
  
  delete[] u;
  delete[] r;
  delete[] h_c;
  delete[] h; 

     
     for (int i = 0; i < dim_hidden_; i++)
     {  
         delete[] Wu[i];
         delete[] Wr[i];
         delete[] Wc[i];
         delete[] Uu[i];
         delete[] Ur[i];
         delete[] Uc[i];
     }

  created_ = 0; 

}

float *GRU::mulMatVec(float **M, float *x, int num_rows, int num_cols)
{
      float *ret = new float[num_rows];
      
        for (int i = 0; i < num_rows; i++)     
         {
           ret[i] = 0.0;
           for (int j = 0; j < num_cols; j++)
           {
               ret[i] += M[i][j] * x[j]; 
           }
         }
      return ret;    
}

void GRU::setParameters(float *theta)
{
           
      for (int i = 0; i < dim_hidden_; i++)     
      {
          for (int j = 0; j < dim_input_; j++)
           {
              Wu[i][j] = *theta++;
              Wr[i][j] = *theta++;
              Wc[i][j] = *theta++;
           }                             
         
          for (int j = 0; j < dim_hidden_; j++)
           {
              Uu[i][j] = *theta++;
              Ur[i][j] = *theta++;
              Uc[i][j] = *theta++;
           }            
              bu[i] = *theta++;                      
              br[i] = *theta++;
              bc[i] = *theta++;
       }        
                 
           
}


float *GRU::getParameters()
{  
   float *theta = new float[getNumParameters()];
   unsigned int cc = 0;
   
   for (int i = 0; i < dim_hidden_; i++)     
   {
          for (int j = 0; j < dim_input_; j++)
           {
              theta[cc++] = Wu[i][j]; 
              theta[cc++] = Wr[i][j];              
              theta[cc++] = Wc[i][j];              
           }
  
          for (int j = 0; j < dim_hidden_; j++)
           {
              theta[cc++] = Uu[i][j];              
              theta[cc++] = Ur[i][j];              
              theta[cc++] = Uc[i][j];                 
           }

             theta[cc++] = bu[i];         
             theta[cc++] = br[i];              
             theta[cc++] = bc[i];
                                              
   }

              
  return theta;
}

int GRU::getNumParameters()
{
  return 3 * dim_hidden_ * (dim_input_ + dim_hidden_ + 1);
}


void GRU::setState(float *state)
{
for (int i = 0 ; i < dim_hidden_; i++)
  h[i] = *state++;
}

float * GRU::getState()
{
   return h; 
}




int GRU::getStateSize()
{
   return dim_hidden_; 
}

int GRU::getInputSize()
{
   return dim_input_; 
}

void GRU::resetState()
{
  for (int i = 0; i < dim_hidden_; i++)
  {
      h[i] = 0.0;
      u[i] = 1.0;
      r[i] = 1.0;
      h_c[i] = 0.0;
  }   
}


void GRU::calculateUpdateGate(float *x)
{
  float *Wux = mulMatVec(Wr, x, dim_hidden_, dim_input_);
  float *Uuh = mulMatVec(Uu, h, dim_hidden_, dim_hidden_);
  for (int i = 0; i < dim_hidden_; i++)
    { 
      u[i] = sigmoid(Wux[i] + Uuh[i] + bu[i]);
    }

  delete[] Wux;
  delete[] Uuh;
}

void GRU::calculateResetGate(float *x)
{
  float *Wrx = mulMatVec(Wr, x, dim_hidden_, dim_input_);
  float *Urh = mulMatVec(Ur, h, dim_hidden_, dim_hidden_);
  for (int i = 0; i < dim_hidden_; i++)
    { 
      r[i] = sigmoid(Wrx[i] + Urh[i] + br[i]);
    }

  delete[] Wrx;
  delete[] Urh;
}

void GRU::calculateCandidateState(float *x)
{
  float *Wcx = mulMatVec(Wc, x, dim_hidden_, dim_input_);
  float *Uch = mulMatVec(Uc, h, dim_hidden_, dim_hidden_);
  
  for (int i = 0; i < dim_hidden_; i++)
    { 
      h_c[i] = th(Wcx[i] + Uch[i] * r[i] + bc[i]);
    }
    
  delete[] Wcx;
  delete[] Uch;
    
}

void GRU::input(float *x)
{
   if (step_ % num_steps_ == 0 && num_steps_ > 0)
        {
            resetState();
        }

   calculateUpdateGate(x);
   calculateResetGate(x);
   calculateCandidateState(x);   
  
   for (int i = 0; i < dim_hidden_; i++)
   {
      h[i] = u[i] * h_c[i] + (1 - u[i])*h[i];
   }         
   
   step_++;
   
}

