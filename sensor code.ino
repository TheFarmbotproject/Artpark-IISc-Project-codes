//DHT11 Sensor
#include "DHT.h"
#define DHTPIN 6
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);




//Light intensity Sensor
#include <Wire.h>
#include <BH1750.h>
BH1750 lightMeter;


//Accelerometer- Gyroscope GY-521
const int MPU = 0x68;
int16_t AcX, AcY, AcZ, Tmp, GyX, GyY, GyZ;

//Magnetometer HMC5883L
#include <Arduino.h>
#include <HMC5883L_Simple.h>
HMC5883L_Simple Compass;

//MQ135
#include <MQUnifiedsensor.h>
#define placa "Arduino MEGA"
#define Voltage_Resolution 5
#define pin A0
#define type "MQ-135"
#define ADC_Bit_Resolution 10
#define RatioMQ135CleanAir 3.6
MQUnifiedsensor MQ135(placa, Voltage_Resolution, ADC_Bit_Resolution, pin, type);

//Soil moisture

const int AirValue = 806;
const int WaterValue = 599;
int soilMoistureValue = 0;
int soilmoisturepercent = 0;

void setup() {
  Serial.begin(9600);
  Wire.begin();

  //DHT11
  dht.begin();


  //Light Sensor
  lightMeter.begin();

  //Accelerometer- Gyroscope GY-521
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  //Magnetometer HMC5883L
  Compass.SetDeclination(-0, 23, 'W');
  Compass.SetSamplingMode(COMPASS_SINGLE);
  Compass.SetScale(COMPASS_SCALE_130);
  Compass.SetOrientation(COMPASS_HORIZONTAL_X_NORTH);

  //MQ135
  MQ135.setRegressionMethod(1);
  MQ135.init();
  float calcR0 = 0;
  for (int i = 1; i <= 10; i ++)
  {
    MQ135.update();
    calcR0 += MQ135.calibrate(RatioMQ135CleanAir);
  }
  MQ135.setR0(calcR0 / 10);
  if (isinf(calcR0)) {
    while (1);
  }
  if (calcR0 == 0) {
    while (1);
  }
}

void loop() {
  //dht11
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (isnan(h) || isnan(t)) {
    return;
  }


  // MQ135 Gas
  MQ135.update();
  MQ135.setA(605.18); MQ135.setB(-3.937);
  float CO = MQ135.readSensor();
  MQ135.setA(77.255); MQ135.setB(-3.18);
  float Alcohol = MQ135.readSensor();
  MQ135.setA(110.47); MQ135.setB(-2.862);
  float CO2 = MQ135.readSensor();
  MQ135.setA(44.947); MQ135.setB(-3.445);
  float Toluen = MQ135.readSensor();
  MQ135.setA(102.2 ); MQ135.setB(-2.473);
  float NH4 = MQ135.readSensor();
  MQ135.setA(34.668); MQ135.setB(-3.369);
  float Aceton = MQ135.readSensor();

  //soil moisture
  soilMoistureValue = analogRead(A1);  //put Sensor insert into soil
  Serial.println(soilMoistureValue);
  soilmoisturepercent = map(soilMoistureValue, AirValue, WaterValue, 0, 100);

  //light sesnor
  float lux = lightMeter.readLightLevel();

  //Accelerometer- Gyroscope GY-521
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 12, true);
  AcX = Wire.read() << 8 | Wire.read();
  AcY = Wire.read() << 8 | Wire.read();
  AcZ = Wire.read() << 8 | Wire.read();
  GyX = Wire.read() << 8 | Wire.read();
  GyY = Wire.read() << 8 | Wire.read();
  GyZ = Wire.read() << 8 | Wire.read();


  //Magnetometer HMC5883L
  float heading = Compass.GetHeadingDegrees();



  Serial.print(h);
  Serial.print(" ");
  Serial.print(t);
  Serial.print(" ");
  Serial.print(CO);
  Serial.print(" ");
  Serial.print(CO2 + 400);
  Serial.print(" ");
  Serial.print(Toluen);
  Serial.print(" ");
  Serial.print(NH4);
  Serial.print(" ");
  Serial.print(soilmoisturepercent);
  Serial.print(" ");
  Serial.print(lux);
  Serial.print(" ");
  Serial.print(AcX);
  Serial.print(AcY);
  Serial.print(AcZ);
  Serial.print(" ");
  Serial.print( heading );
  Serial.print(" ");

  delay(1000);

}
