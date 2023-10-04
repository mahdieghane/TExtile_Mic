/**
 * Simple Read
 * 
 * Read data from the serial port and change the color of a rectangle
 * when a switch connected to a Wiring or Arduino board is pressed and released.
 * This example works with the Wiring / Arduino program that follows below.
 */


import processing.serial.*;
import processing.net.*; 
    
Serial myPort1;  // Create obje end-of-stream.ct from Serial class
Client myClient1; 
int dataStreaming1 = 1; // 0: from serial port, 1: from socket server
int featureRowNum1 = 256;
int featureColumnNum1 = 256;
float maximumFeatureValue1 = 0.3;
int index1 = 0;
Float[][] features1 = new Float[featureRowNum1][featureColumnNum1];      // Data received from the serial port
boolean isStreaming1 = false;
PImage spetrogramImage1;
String[] objects1 = {"Tap", "Swipe", "Knock", "Slap", "Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping" , "Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Dispensing Tape", "Grating"};
int recordingTime1 = 0;
int[] values1;
String result1 = "";
float graphRatio1 = 0.3;
boolean showValue1 = false;
int visualizationStyle1 = 2; // 0: line, 1:feature bar chart, 2:spectrogram
void setup1() 
{
  size(500,500);
  frameRate(30);
  spetrogramImage1 = createImage(featureRowNum1, featureColumnNum1, RGB);

  if(dataStreaming1 == 0){
   
    try{
      printArray(Serial.list());
      //String portName = Serial.list()[4];
      myPort1 = new Serial(this, "/dev/cu.usbmodem1421401", 115200);
      myPort1.bufferUntil('\n'); 
      //dataStreaming = 0;
    }
    catch(Exception e){
      println("Serial Port Error");
     }
  }  
  else {
     myClient1 = new Client(this, "localhost", 8081); 
     if(myClient1 != null) 
       dataStreaming1 = 1;
  }
  for(int i = 0 ; i < featureRowNum1; i++){
    for(int j = 0 ; j < featureColumnNum1; j++){
      features1[i][j] = -1f;
    }
  }
 resetText1();
  //result1 = "Hello";
  values1 = new int[width];
  print("Start, Data Streaming Type:");
  print(dataStreaming1);
  noSmooth();
}

void resetText1(){
   result1 = "None";
   for(int i = 0 ; i < objects1.length;i++){
    result1 += "(" + (char)(i+'A') + ")  " + objects1[i] + ", ";
  }
}

void draw1()
{

  if(dataStreaming1 == 1) {
    readFromSocket1();
  }
  background(0);             // Set background to black
  
  
  if(visualizationStyle1 == 0){
      float graphY = height * (1-graphRatio1);
      float graphHeight = height * graphRatio1;

      drawLines1(graphY, graphHeight);                   // set fill to black
  } else if (visualizationStyle1 == 1){
      int rectWidth = (width-100) / featureColumnNum1;
      for(int i = 0; i < featureColumnNum1; i++){
         fill(1, 166, 240);
         if(features1[index1][i] > 0)
             rect(50+rectWidth*i, 800, rectWidth, -features1[index1][i]*2);
      }
  } else if (visualizationStyle1 == 2){
      colorMode(HSB, 1);
      spetrogramImage1.loadPixels();
      int columnIndex = 0;
      for(int i = index1 ; i < featureColumnNum1; i ++){
        for(int j = 0; j < featureRowNum1; j ++){
              float value = features1[i][j] < 0 ? 0:features1[i][j]/featureColumnNum1;
              spetrogramImage1.pixels[columnIndex+(featureRowNum1-j-1)*featureColumnNum1] = color((0.4*value+0.6), 1, value);
        }
        columnIndex++;
      }
      for(int i =  0; i < index1&& columnIndex<featureColumnNum1; i ++){
        for(int j = 0; j < featureRowNum1; j ++){
              float value = features1[i][j] < 0 ? 0:features1[i][j]/featureColumnNum1;
              spetrogramImage1.pixels[columnIndex+(featureRowNum1-j-1)*featureColumnNum1] = color((0.4*value+0.6), 1, value);
              
        }
        columnIndex++;
      }
      spetrogramImage1.updatePixels();
      image(spetrogramImage1, 0, height/5+160, width, height/5-40);
  }
  //if(!isStreaming1){
  //  for (int i = 0; i < featureRowNum1; i++){
  //       features[index][i] = -1f;
  //  }
  //  index = (++index)%featureColumnNum1;
  //}
  //isStreaming1 = false;  
  colorMode(RGB, 255);
  fill(255);
  textSize(10);
  textAlign(CENTER, CENTER);
  //result1 = "hello";
  //println(result1);
  text(result1, width/2, 400);
  
  if(recordingTime1 != 0){
    //print(recordingTime1 - millis());
    if(millis() - recordingTime1 > 3000){
      resetText1();
      recordingTime1 = 0;
    }
  }
  updatePixels();

 
}

void keyPressed1() {
   if(dataStreaming1==1 && myClient1 != null){
       char c = Character.toUpperCase(key);
       if((int)c >= 'A' && (int)c <= 'Z'){
         myClient1.write(' ');
         myClient1.write(c);
         myClient1.write(' ');
         println(c);
       }
    }
    
   int index1 = Character.toUpperCase(key) - 'A';
   if(index1 >= 0 && index1 < objects1.length)
     result1 = objects1[index1];
  if (key == 'r' || key == 'R') {
     result1 += " recording";
     recordingTime1 = millis();
  } else if (key == 'x' || key == 'X') {
     result1 += " overwrite";
  }else if (key == 's' || key == 'S') {
      resetText1();
      recordingTime1 = 0;
  }
  //} else if (key == 'a' || key == 'A') {
     
  //    if(dataStreaming1==1 && myClient != null){
  //       myClient.write(" A ");
  //    }
  //} else if (key == 's' || key == 'S') {
  //    if(dataStreaming1==1 && myClient != null){
  //       myClient.write(" S ");
  //    }
  //} 
}

void readFromSocket1(){

    if ( myClient1.available() > 0) { 
      String data = myClient1.readStringUntil('\n');
      if(data == null) return;
      data = data.trim();
      if(data.indexOf(',') != -1){
          String[] dataArray = data.split(",");
          
          if(dataArray[0].equals("feature")){
              for (int i = 0; i < featureRowNum1; i++){
                if(i < dataArray.length) 
                   features1[index1][i] = (Float.valueOf(dataArray[i+1]));
                else
                   features1[index1][i] = -1f;
              }
              index1 = (++index1)%featureColumnNum1;
          }
          if(dataArray[0].equals("data")){
              for (int i = 1; i < dataArray.length; i++){
                   pushValue1(int(Float.valueOf(dataArray[i])*5000));
              }
              
              //val = Integer.valueOf(dataArray[1]);
          }
          if(dataArray[0].equals("result1")){
              result1 = "";
              for (int i = 1; i < dataArray.length; i++){
                   if(i!=1) result1 += ",";
                   result1 += dataArray[i].trim();
              }
          }
      }
    }
  
}


void readFromSerial1(){
    //result1 = "Noise";
    if ( myPort1.available() > 0) { 
      String data = myPort1.readStringUntil('\n');  
      //print(data);
      if(data.indexOf(',') != -1){
          String[] dataArray = data.split(",");
          if(dataArray[0].equals("feature")){
              for (int i = 1; i < featureRowNum1; i++){
                try{
                  //print(dataArray.length);
                  if(i < dataArray.length) 
                     features1[index1][i] = (Float.valueOf(dataArray[i]));
                  else
                     features1[index1][i] = -1f;
                }
                catch(NumberFormatException e){
                   print("Parse Error");
                    break;
                }
              }
              index1 = (++index1)%featureColumnNum1;
              isStreaming1 = true;
          }
          if(dataArray[0].equals("data")){
              for (int i = 1; i < dataArray.length; i++){
                   try {
                       pushValue1(Integer.valueOf(dataArray[i]));
                   } catch (NumberFormatException e) {
                      System.out.println("Input String cannot be parsed to Integer.");
                  }
              }
              //val = Integer.valueOf(dataArray[1]);
          }
          if(dataArray[0].equals("result1")){
              result1 = dataArray[1].trim();
          }
      }
    }
  
}

 
void serialEvent1(Serial p) { 
  readFromSerial1(); 
}

void pushValue1(int value) {
  for (int i=0; i<width-1; i++)
    values1[i] = values1[i+1];
  values1[width-1] = value;
}

void drawLines1(float graphY, float graphHeight) {

  //int displayWidth = (int) (width / zoom);
  stroke(1, 166, 240);
  strokeWeight(5);
  int k = values1.length - width;

  float x0 = 0;
  float y0 = graphY - values1[k];
  for (int i=1; i<width; i++) {
    k++;
    float x1 = (int) i ;
    float y1 = graphY - values1[k];
    line(x0, y0, x1, y1);
    x0 = x1;
    y0 = y1;
  }
}
