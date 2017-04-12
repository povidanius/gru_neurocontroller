// Gated recurrent unit (GRU) recurrent neural network neurocontroller
// GRU can be trained externally e.g. on GPU
// and uploaded via GRU::setParameters() method

// code by Povilas Daniusis
// povilas.daniusis@gmail.com
// 


class GRU {
private:
  float **Wu, **Uu, *bu;
  float **Wr, **Ur, *br;
  float **Wc, **Uc, *bc;
  float *h;
  int dim_input_;
  int dim_hidden_;
  
  float* mulMatVec(float **M, float *x, int num_rows, int num_cols);
  float* addVectors(float *x, float *y, int dim);
  float* elementwiseMulVectors(float *x, float *y, int dim);
  float* scaleShiftVector(float *x, float scale, float shift, int dim);
  
  
  float sigmoidScalar(float x) { return (float) 1.0/(1.0 + exp(-x)); };
  float *sigmoidVector(float *x, int dim) { float *ret = new float[dim]; for (int i = 0; i < dim; i++) ret[i] = sigmoidScalar(x[i]); return ret; }
  
  float tanh(float x){  return   (float) (exp(x) - exp(-x)) /(exp(x) + exp(-x)); };
  float *tanhVector(float *x, int dim) { float *ret = new float[dim]; for (int i = 0; i < dim; i++) ret[i] = sigmoidScalar(x[i]); return ret; }
  
  int created_;
  int active_;
  long step_;
  
public:
 GRU(int dim_input, int dim_hidden);
 ~GRU();
 void setParameters(float *theta);
 float *getParameters();
 int getNumParameters();
 int getStateSize();
 int getInputSize();
 int getStatus();
 void setStatus(int status);
 
 
 void setState(float *theta); 
 void resetState(); 
 void input(float *input);
 float *getState(); 
};

GRU::GRU(int dim_input, int dim_hidden) : created_(0), active_(0), step_(0)
{
   dim_input_ = dim_input;
   dim_hidden_ = dim_hidden;
   
   Wu = new float *[dim_hidden];   
   Wr = new float *[dim_hidden];   
   Wc = new float *[dim_hidden];
   
   Uu = new float *[dim_hidden];   
   Ur = new float *[dim_hidden];   
   Uc = new float *[dim_hidden];
   
   for (int i = 0; i < dim_input; i++)
     {
       Wu[i] = new float[dim_input];
       Wr[i] = new float[dim_input];
       Wc[i] = new float[dim_input];
     }
   for (int i = 0; i < dim_hidden; i++)
     {  
       Uu[i] = new float[dim_input];
       Ur[i] = new float[dim_input];
       Uc[i] = new float[dim_input];
     }  
    bu = new float[dim_hidden];
    br = new float[dim_hidden];
    bc = new float[dim_hidden];
    h = new float[dim_hidden];
    
    resetState();
    
    created_ = 1;       
}

GRU::~GRU()
{
  delete[] bu;
  delete[] br;
  delete[] bc;
  delete[] h;
  
     for (int i = 0; i < dim_input_; i++)
     {
         delete[] Wu[i];
         delete[] Wr[i];
         delete[] Wc[i];
     } 
     
     for (int i = 0; i < dim_hidden_; i++)
     {  
         delete[] Uu[i];
         delete[] Ur[i];
         delete[] Uc[i];
     }
  
}

float *GRU::mulMatVec(float **M, float *x, int num_rows, int num_cols)
{
      float *ret = new float[num_rows];
      
        for (int i = 0; i < num_rows; i++)     
         {
           ret[i] = 0.0;
          for (int j = 0; j < num_cols; j++)
           {
               ret[i] += M[j][i] * x[i];
           }
         }
      return ret;    
}

float *GRU::addVectors(float *x, float *y, int dim)
{
  float *z = new float[dim];
  for (int i = 0; i < dim; i++)
   z[i] = x[i] + y[i];
   return z;
  
}

float *GRU::elementwiseMulVectors(float *x, float *y, int dim)
{
  float *ret = new float[dim];
  
  for (int i = 0; i < dim; i++)
  {
      ret[i] = x[i] * y[i];    
  } 
  
  return ret;
}

float *GRU::scaleShiftVector(float *x, float scale, float shift, int dim)
{
  float *ret;  
  
  for (int i = 0 ; i < dim; i++)  
    ret[i] = scale * x[i] + shift;
    
  return ret;      
}




void GRU::setParameters(float *theta)
{
  
      for (int i = 0; i < dim_input_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              Wu[i][j] = *theta++;
           }
         for (int i = 0; i < dim_input_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              Wr[i][j] = *theta++;
           }    
           
         for (int i = 0; i < dim_input_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              Wc[i][j] = *theta++;
           }    
           
            for (int i = 0; i < dim_hidden_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              Uu[i][j] = *theta++;
           }
         for (int i = 0; i < dim_hidden_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              Ur[i][j] = *theta++;
           }    
           
         for (int i = 0; i < dim_hidden_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              Uc[i][j] = *theta++;
           }    

        for (int i = 0; i < dim_hidden_; i++)                        
              bu[i] = *theta++;    
        for (int i = 0; i < dim_hidden_; i++)                        
              br[i] = *theta++;     
        for (int i = 0; i < dim_hidden_; i++)                        
              bc[i] = *theta++;                
           
}

int GRU::getNumParameters()
{
  return 3 * dim_hidden_ * (dim_input_ + dim_hidden_ + 1);
}

float *GRU::getParameters()
{  
  float *theta = new float[getNumParameters()];
  
   for (int i = 0; i < dim_input_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              *theta = Wu[i][j];
              theta++;
           }
         for (int i = 0; i < dim_input_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              *theta = Wr[i][j];
              theta++;
           }    
           
         for (int i = 0; i < dim_input_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              Wc[i][j] = *theta;
              theta++;
           }    
           
            for (int i = 0; i < dim_hidden_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              *theta = Uu[i][j];
              theta++;
           }
         for (int i = 0; i < dim_hidden_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
             *theta = Ur[i][j];
             theta++;
           }    
           
         for (int i = 0; i < dim_hidden_; i++)     
          for (int j = 0; j < dim_hidden_; j++)
           {
              *theta = Uc[i][j];
              theta++;
           }    

        for (int i = 0; i < dim_hidden_; i++, theta++)                        
        {
              *theta = bu[i];
              
        }
        for (int i = 0; i < dim_hidden_; i++, theta++)                        
        {
              *theta = br[i] ;     
        }
        for (int i = 0; i < dim_hidden_; i++, theta++)                        
        {
            
              *theta = bc[i];
        }
             
              
  return theta;
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

int GRU::getStatus()
{
  return active_; 
}

void GRU::setStatus(int status)
{
  active_ = status; 
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
      h[i] = 0.0;
}

void GRU::input(float *x)
{
   float *u =  addVectors( addVectors( mulMatVec(Wu, x, dim_hidden_, dim_input_) , mulMatVec(Uu, h, dim_hidden_, dim_hidden_), dim_hidden_), bu, dim_hidden_);
   u = sigmoidVector(u, dim_hidden_);
   
   float *r =  addVectors( addVectors( mulMatVec(Wr, x, dim_hidden_, dim_input_) , mulMatVec(Ur, h, dim_hidden_, dim_hidden_), dim_hidden_), br, dim_hidden_);
   r = sigmoidVector(r, dim_hidden_);
   
   float *c =  mulMatVec(Wc, x, dim_hidden_, dim_input_);
   c = addVectors(c,  elementwiseMulVectors( mulMatVec(Uc, x, dim_hidden_, dim_input_), r, dim_hidden_), dim_hidden_);
   c = addVectors(c, bc, dim_hidden_);  
   c = tanhVector(c, dim_hidden_);
   
   
   float *negu = scaleShiftVector(u, -1.0, 1.0, dim_hidden_);
   h = addVectors( elementwiseMulVectors(h, u, dim_hidden_), elementwiseMulVectors(c, negu, dim_hidden_), dim_hidden_);   
}

//#define HWSERIAL Serial1

#define INPUT_DIM 4
#define HIDDEN_DIM 34
#define OUTPUT_DIM 1

int INPUT_PINS[] = {A2, A3, A10, A11};
int OUTPUT_PINS[] = {A13};
float input_data[INPUT_DIM];


GRU gru(INPUT_DIM, HIDDEN_DIM);


void resetState()
{
   gru.resetState(); 
}

void uploadParameters()
{
  int num_params = gru.getNumParameters();
    
  float *x = new float[num_params];
   for (int i = 0; i < num_params; i++)
   {
     if (Serial.available())    
       x[i] = Serial.parseFloat();       
   } 
  
  gru.setParameters(x); 
  delete[] x;  
  
  
}

void inputData()
{
  int input_size = gru.getInputSize();
  float *x = new float[input_size];
   for (int i = 0; i < input_size; i++)
   {
     if (Serial.available())    
       x[i] = Serial.parseFloat();
       
   } 
  
  gru.input(x);  
  delete[] x;  
}


void outputState()
{
   float *h = gru.getState();
   for (int i = 0; i < gru.getStateSize(); i++)
   {
      Serial.print(h[i], DEC); 
      Serial.print(" ");
   }
   Serial.println();
   delete[] h;
   
}

void activate()
{
   if (gru.getStatus() == 0)
   {
     gru.setStatus(1);
   }
  else
  {
     gru.setStatus(0);
  } 
     
}


void parseSerial() {
  char c = Serial.read();

  switch (c) {
  case 'u': 
    uploadParameters();
    break;
  case 'i': 
    inputData();
    break;
  case 'o': 
    outputState();
    break;  
  case 'r':
    resetState();
  break;  
  case 'a':
    activate();
  break;
  default:  
  break;
  }
}



void setup() {
  
        gru.setStatus(0);
        
	Serial.begin(115200);
  
        for (int i = 0; i < INPUT_DIM; i++)
        {
         pinMode(INPUT_PINS[i], INPUT); 
        } 
        
        for (int i = 0; i < OUTPUT_DIM; i++)
        {
          pinMode(OUTPUT_PINS[i], OUTPUT); 
        }       
}


void loop() {  
  
  if (gru.getStatus())
  {
    for (int i = 0; i < INPUT_DIM; i++)
    {
      input_data[i] = (float) analogRead(INPUT_PINS[i]);
    }  
  
    gru.input(input_data);
    float *h = gru.getState();
    
 /*   for (int i = 0; i < OUTPUT_DIM; i++)
    {
        analogWrite(OUTPUT_PINS[i], h[i]);
    }*/
    
    delete[] h;
  }  
  
  
  if (Serial.available()) parseSerial();


}



