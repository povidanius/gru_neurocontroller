
#include <SD.h>
#include <SPI.h>

#define SIGMOID(x) 

class GRU {
private:
  float *h; // state
  float *u; // update gate
  float *r; // reset gate
  float *h_c; // candidate state

  float **Wu, **Uu, *bu;
  float **Wr, **Ur, *br;
  float **Wc, **Uc, *bc;  
  
  int dim_input_;
  int dim_hidden_;
  
  float* mulMatVec(float **M, float *x, int num_rows, int num_cols); 
  
  void calculateUpdateGate(float *x);
  void calculateResetGate(float *x);
  void calculateCandidateState(float *x);
  
  float sigmoid(float x) { return (float) 1.0/(1.0 + exp(-x)); };
  
  float th(float x){  return   (float) (exp(x) - exp(-x)) /(exp(x) + exp(-x)); };
  
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
 long getStep() { return step_; }
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
      theta[i] = (float) 0.1;
      
    setParameters(theta);
    
    delete[] theta;
    
    
    created_ = 1;     
      
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
  
}

float *GRU::mulMatVec(float **M, float *x, int num_rows, int num_cols)
{
      float *ret = new float[num_rows];
      
        for (int i = 0; i < num_rows; i++)     
         {
           ret[i] = 0.0;
          for (int j = 0; j < num_cols; j++)
           {
               ret[i] += M[i][j] * x[i];
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

   calculateUpdateGate(x);
   calculateResetGate(x);
   calculateCandidateState(x);   
  
   for (int i = 0; i < dim_hidden_; i++)
   {
      h[i] = u[i] * h_c[i] + (1 - u[i])*h[i];
   }   
      

   step_++;
}


#define INPUT_DIM 5
#define HIDDEN_DIM 5
#define OUTPUT_DIM 1

GRU gru(INPUT_DIM, HIDDEN_DIM);



// On the Ethernet Shield, CS is pin 4. Note that even if it's not
// used as the CS pin, the hardware CS pin (10 on most Arduino boards,
// 53 on the Mega) must be left as an output or the SD library
// functions will not work.

// change this to match your SD shield or module;
// Arduino Ethernet shield: pin 4
// Adafruit SD shields and modules: pin 10
// Sparkfun SD shield: pin 8
// Teensy audio board: pin 10
// Teensy 3.5 & 3.6 on-board: BUILTIN_SDCARD
// Wiz820+SD board: pin 4
// Teensy 2.0: pin 0
// Teensy++ 2.0: pin 20
const int chipSelect =  BUILTIN_SDCARD;
long int c = 0;

void setup()
{
  //UNCOMMENT THESE TWO LINES FOR TEENSY AUDIO BOARD:
  //SPI.setMOSI(7);  // Audio shield has MOSI on pin 7
  //SPI.setSCK(14);  // Audio shield has SCK on pin 14

  
 // Open serial communications and wait for port to open:
  Serial.begin(9600);

  Serial.print("Initializing SD card...");
  
  // see if the card is present and can be initialized:
  if (!SD.begin(chipSelect)) {
    Serial.println("Card failed, or not present");
    // don't do anything more:
    return;
  }
  Serial.println("card initialized.");
/*  Serial.println("Reading parameters");

   // open the file. note that only one file can be open at a time,
  // so you have to close this one before opening another.
  File dataFile = SD.open("parameters.txt");

  // if the file is available, write to it:
  if (dataFile) {
    int numParameters = gru.getNumParameters();
    for (int i = 0; i < gru.getNumParameters(); i++)
    {
      if (dataFile.available())
      {
        dataFile.read()
      }      
    }
    //}
    //while (dataFile.available()) {
    //      Serial.write(dataFile.read());
    //}
    dataFile.close();
  }  */

  
  
  randomSeed(analogRead(0));
  gru.setStatus(1);
  Serial.print("Num parameters = ");
  Serial.println(gru.getNumParameters());
  
  float *theta = gru.getParameters();
  for (int i = 0; i < gru.getNumParameters(); i++)
  { 
    Serial.print(i);
    Serial.print(" ");  
    Serial.println(theta[i]);    
  }
  delete[] theta;
  

  Serial.println();
  
  
}


void loop()
{

    float input_data[INPUT_DIM];
    //Serial.print(gru.getStep());
    //Serial.print(" ");
    
    for (int i = 0; i < INPUT_DIM; i++)
    {
      input_data[i] = (float) random(300)/ 299;       
     // Serial.print(input_data[i]);
      //Serial.print(" ");
    }  
    //Serial.println();
    
  
    gru.input(input_data);

    float *h = gru.getState();    
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      Serial.print(h[i]);
      Serial.print(" ");
    }
    Serial.println();   
    

    
    
    
 /* }
  else
  {
    Serial.println("Waiting ... ");
  }
  */

  
  /*
  // make a string for assembling the data to log:
  String dataString = "";

  // read three sensors and append to the string:
  for (int analogPin = 0; analogPin < 3; analogPin++) {
    int sensor = analogRead(analogPin);
    dataString += String(sensor);
    if (analogPin < 2) {
      dataString += ","; 
    }
  }

  // open the file. note that only one file can be open at a time,
  // so you have to close this one before opening another.
  File dataFile = SD.open("datalog.txt", FILE_WRITE);

  // if the file is available, write to it:
  if (dataFile) {
    dataFile.println(dataString);
    dataFile.close();
    // print to the serial port too:
    Serial.println(dataString);
  }  
  // if the file isn't open, pop up an error:
  else {
    Serial.println("error opening datalog.txt");
  } */

 // Serial.println();
  
}









