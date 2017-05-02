
#include <SD.h>
#include <SPI.h>
#include <GRU.h>

#define N_STEPS  0
#define INPUT_DIM 4
#define HIDDEN_DIM 8
#define OUTPUT_DIM 1
GRU gru(INPUT_DIM, HIDDEN_DIM, N_STEPS);

long int c = 0;

void readParameters()
{
  Serial.println("Reading parameters");
  File dataFile = SD.open("parameters.txt");
  if (dataFile) {
    int numParameters = gru.getNumParameters();
    float *theta = new float[numParameters];
    for (int i = 0; i < gru.getNumParameters(); i++)
    {
      if (dataFile.available())
      {
            theta[i] = dataFile.parseFloat();
      }      
      else
      {
            Serial.print("Parameter file is corrupted.");
            return;
      }
    }
    gru.setParameters(theta);
    dataFile.close();
  }    
}

void printParameters()
{
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

void setup()
{ 

  Serial.begin(9600);
  randomSeed(analogRead(0));

  Serial.print("Initializing SD card..."); 
  if (!SD.begin(BUILTIN_SDCARD)) {
    Serial.println("Card failed, or not present");
    return;
  }
  Serial.println("Card initialized.");
  readParameters();
  printParameters();
  
  
}


void loop()
{
    float input_data[INPUT_DIM];
    //Serial.print(gru.getStep());
    //Serial.print(" ");
    
    for (int i = 0; i < INPUT_DIM; i++)
    {
      input_data[i] = gru.randn(0.0, 1.0); //analogRead(i)/1024.0;      
     // Serial.print(input_data[i]);
      //Serial.print(" ");
    }  
    //Serial.println();
    
  
    gru.input(input_data);
    //Serial.println(analogRead(0));
    

    /*float *h = gru.getState();    
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
      Serial.print(h[i]);
      Serial.print(" ");
    }
    Serial.println();      
     */   

}









