#include <SoftwareSerial.h>
SoftwareSerial mySerial(10, 11); // RX, TX

//Ultrasonic_Sensor_4
#define echoPin_4  29// attach pin D2 Arduino to pin Echo of HC-SR04
#define trigPin_4 27 //attach pin D3 Arduino to pin Trig of HC-SR04

//Ultrasonic_Sensor_3
#define echoPin_3  25// attach pin D2 Arduino to pin Echo of HC-SR04
#define trigPin_3 23 //attach pin D3 Arduino to pin Trig of HC-SR04

//Ultrasonic_Sensor_1
#define echoPin_1  24// attach pin D2 Arduino to pin Echo of HC-SR04
#define trigPin_1 22 //attach pin D3 Arduino to pin Trig of HC-SR04

// defines variables
long duration; // variable for the duration of sound wave travel
int distance; // variable for the distance measurement

char Motor_inputs[8];
String Motor_input;

#include <Servo.h>


Servo ESC1;
Servo ESC2;
int potValue1;
int potValue2;
Servo myservo;
int pos = 0;

int Lswitch1 = 50;
int Lswitch2 = 52;
int flag1 = 0;
int flag2 = 0;
void setup() {

  myservo.attach(18);
  pinMode(trigPin_4, OUTPUT); // Sets the trigPin as an OUTPUT
  pinMode(echoPin_4, INPUT); // Sets the echoPin as an INPUT

  pinMode(trigPin_3, OUTPUT); // Sets the trigPin as an OUTPUT
  pinMode(echoPin_3, INPUT); // Sets the echoPin as an INPUT

  pinMode(trigPin_1, OUTPUT); // Sets the trigPin as an OUTPUT
  pinMode(echoPin_1, INPUT); // Sets the echoPin as an INPUT

  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);

  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT);
  pinMode(16, OUTPUT);
  pinMode(15, OUTPUT);
  pinMode(Lswitch1, INPUT);
  pinMode(Lswitch2, INPUT);


  Serial.begin(57600);
  while (!Serial) {
    ;
  }
  mySerial.begin(9600);
}

void loop() { // run over and over

  if ( (digitalRead(Lswitch1) == LOW) && (flag1 == 0) )
  {
    flag1 = 1;
    delay(20);
  }

  if ( (digitalRead(Lswitch1) == HIGH) && (flag1 == 1) )
  {
    flag1 = 0;
    delay(20);
  }


  if ( flag1 == 1 )
  {
    digitalWrite(2, HIGH);
    digitalWrite(3, HIGH);
    digitalWrite(4, HIGH);
    digitalWrite(5, HIGH);

  }

  digitalWrite(Lswitch1, HIGH);


  if ( (digitalRead(Lswitch2) == LOW) && (flag2 == 0) )
  {
    flag2 = 1;
    delay(20);
  }

  if ( (digitalRead(Lswitch2) == HIGH) && (flag2 == 1) )
  {
    flag2 = 0;
    delay(20);
  }


  if ( flag2 == 1 )
  {
    digitalWrite(2, HIGH);
    digitalWrite(3, HIGH);
    digitalWrite(4, HIGH);
    digitalWrite(5, HIGH);

  }

  digitalWrite(Lswitch2, HIGH);

  //-----------------------------------Ultrasonic-4---------------------------------------
  digitalWrite(trigPin_4, LOW);
  delayMicroseconds(2);
  // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
  digitalWrite(trigPin_4, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin_4, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin_4, HIGH);
  // Calculating the distance
  distance = duration * 0.034 / 2; // Speed of sound wave divided by 2 (go and back)
  // Displays the distance on the Serial Monitor
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");
  if (distance <= 30)
  {
    ESC1.detach();
    ESC2.detach();
  }

  //-----------------------------------Ultrasonic-3---------------------------------------
  digitalWrite(trigPin_3, LOW);
  delayMicroseconds(2);
  // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
  digitalWrite(trigPin_3, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin_3, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin_3, HIGH);
  // Calculating the distance
  distance = duration * 0.034 / 2; // Speed of sound wave divided by 2 (go and back)
  // Displays the distance on the Serial Monitor
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");
  if (distance <= 30)
  {

    ESC1.detach();
    ESC2.detach();
  }

  //-----------------------------------Ultrasonic-1---------------------------------------
  digitalWrite(trigPin_1, LOW);
  delayMicroseconds(2);
  // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
  digitalWrite(trigPin_1, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin_1, LOW);
  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin_1, HIGH);
  // Calculating the distance
  distance = duration * 0.034 / 2; // Speed of sound wave divided by 2 (go and back)

  if (distance <= 30)
  {
    ESC1.detach();
    ESC2.detach();
  }

  delay(1000);
  //-----------------------------------------------------------------------------------------

  if (mySerial.available())
  {

    Motor_input = mySerial.readString();
    Motor_input = Motor_input.substring(0, 6);
  }
  for (int i = 0; i < 6; i++)
  {
    Motor_inputs[i] = Motor_input[i];
  }
  //hoper
  if (!strcmp(Motor_inputs, "Stop00"))
  {
    digitalWrite(16, LOW);
  }
  else if (!strcmp(Motor_inputs, "Front0"))
  {
    digitalWrite(16, HIGH);
  }

  //pump
  if (!strcmp(Motor_inputs, "pump00"))
  {
    digitalWrite(15, LOW);
  }
  else if (!strcmp(Motor_inputs, "pump01"))
  {
    digitalWrite(15, HIGH);
  }
  //rack & penion
  else if (!strcmp(Motor_inputs, "Front1"))
  {
    digitalWrite(2, LOW);
    digitalWrite(3, HIGH);
    digitalWrite(4, LOW);
    digitalWrite(5, HIGH);

  }
  else if (!strcmp(Motor_inputs, "Back01"))
  {
    digitalWrite(2, HIGH);
    digitalWrite(3, LOW);
    digitalWrite(4, HIGH);
    digitalWrite(5, LOW);
  }

  else if (!strcmp(Motor_inputs, "Stop01"))
  {
    digitalWrite(2, HIGH);
    digitalWrite(3, HIGH);
    digitalWrite(4, HIGH);
    digitalWrite(5, HIGH);
  }

  //linear actuator
  else if (!strcmp(Motor_inputs, "Front2"))
  {
    digitalWrite(7, LOW);
    digitalWrite(6, HIGH);

  }
  else if (!strcmp(Motor_inputs, "Back02"))
  {
    digitalWrite(7, HIGH);
    digitalWrite(6, LOW);
  }
  else if (!strcmp(Motor_inputs, "Stop02"))
  {
    digitalWrite(7, HIGH);
    digitalWrite(6, HIGH);
  }
  else if (!strcmp(Motor_inputs, "Rcam04"))
  {
    ESC1.detach();
    ESC2.detach();
    myservo.write(0);
    //delay(1000);

  }
  else if (!strcmp(Motor_inputs, "Lcam04"))
  {
    ESC1.detach();
    ESC2.detach();
    myservo.write(180);
    //delay(1000);

  }
  else if (!strcmp(Motor_inputs, "Midpo4"))
  {

    myservo.write(90);
    //delay(1000);

  }
  else if (!strcmp(Motor_inputs, "Front3"))
  {
    ESC1.attach(21, 1000, 2000);
    ESC2.attach(20, 1000, 2000);
    potValue1 = 700;   // reads the value of the potentiometer (value between 0 and 1023)
    potValue1 = map(potValue1, 0, 1023, 0, 180);   // scale it to use it with the servo library (value between 0 and 180)
    ESC1.write(potValue1);    // Send the signal to the ESC

    potValue2 = 700;   // reads the value of the potentiometer (value between 0 and 1023)
    potValue2 = map(potValue2, 0, 1023, 0, 180);   // scale it to use it with the servo library (value between 0 and 180)
    ESC2.write(potValue2);    // Send the signal to the ESC
  }
  else if (!strcmp(Motor_inputs, "Back03"))
  {
    ESC1.attach(21, 1000, 2000);
    ESC2.attach(20, 1000, 2000);
    potValue1 = 700;   // reads the value of the potentiometer (value between 0 and 1023)
    potValue1 = map(potValue1, 0, 1023, 180, 0);   // scale it to use it with the servo library (value between 0 and 180)
    ESC1.write(potValue1);    // Send the signal to the ESC

    potValue2 = 700;   // reads the value of the potentiometer (value between 0 and 1023)
    potValue2 = map(potValue2, 0, 1023, 180, 0);   // scale it to use it with the servo library (value between 0 and 180)
    ESC2.write(potValue2);    // Send the signal to the ESC
  }
  else if (!strcmp(Motor_inputs, "Right3"))
  {
    ESC1.attach(21, 1000, 2000);
    ESC2.attach(20, 1000, 2000);
    potValue1 = 700;   // reads the value of the potentiometer (value between 0 and 1023)
    potValue1 = map(potValue1, 0, 1023, 180, 0);   // scale it to use it with the servo library (value between 0 and 180)
    ESC1.write(potValue1);    // Send the signal to the ESC

    potValue2 = 700;   // reads the value of the potentiometer (value between 0 and 1023)
    potValue2 = map(potValue2, 0, 1023, 0, 180);   // scale it to use it with the servo library (value between 0 and 180)
    ESC2.write(potValue2);    // Send the signal to the ESC
  }
  else if (!strcmp(Motor_inputs, "Left03"))
  {
    ESC1.attach(21, 1000, 2000);
    ESC2.attach(20, 1000, 2000);
    potValue1 = 700;   // reads the value of the potentiometer (value between 0 and 1023)
    potValue1 = map(potValue1, 0, 1023, 0, 180);   // scale it to use it with the servo library (value between 0 and 180)
    ESC1.write(potValue1);    // Send the signal to the ESC

    potValue2 = 700;   // reads the value of the potentiometer (value between 0 and 1023)
    potValue2 = map(potValue2, 0, 1023, 180, 0);   // scale it to use it with the servo library (value between 0 and 180)
    ESC2.write(potValue2);    // Send the signal to the ESC
  }
  else if (!strcmp(Motor_inputs, "Stop03"))
  {
    ESC1.detach();
    ESC2.detach();
  }


}
