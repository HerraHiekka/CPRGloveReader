#include <Adafruit_LSM6DS33.h>

int sensorPin1 = A1;    // select the input pin for the pressure sensor
int sensorValue1 = 0;  // variable to store the value coming from the sensor

int sensorPin2 = A2;    // select the input pin for the pressure sensor
int sensorValue2 = 0;  // variable to store the value coming from the sensor

int sensorPin3 = A3;    // select the input pin for the pressure sensor
int sensorValue3 = 0;  // variable to store the value coming from the sensor

int echoPin = 12;
int trigPin = 10;
long duration, cm;

Adafruit_LSM6DS33 lsm6ds33;

void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  if (!lsm6ds33.begin_I2C()) {
    // if (!lsm6ds33.begin_SPI(LSM_CS)) {
    // if (!lsm6ds33.begin_SPI(LSM_CS, LSM_SCK, LSM_MISO, LSM_MOSI)) {
    Serial.println("Failed to find LSM6DS33 chip");
    while (1) {
      delay(10);
    }
  }
  lsm6ds33.configInt1(false, false, true); // accelerometer DRDY on INT1
  lsm6ds33.configInt2(false, true, false); // gyro DRDY on INT2
}

void loop() {

  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  lsm6ds33.getEvent(&accel, &gyro, &temp);

  // read the value from the sensor:
  sensorValue1 = analogRead(sensorPin1); //reads the sensor, its is analog range is from 0 to 1023
  sensorValue2 = analogRead(sensorPin2); //reads the sensor, its is analog range is from 0 to 1023
  sensorValue3 = analogRead(sensorPin3); //reads the sensor, its is analog range is from 0 to 1023

  Serial.print(millis());
  Serial.print(",");
  Serial.print(sensorValue1); // prints the value on the serial monitor
  Serial.print(",");
  Serial.print(sensorValue2); // prints the value on the serial monitor
  Serial.print(",");
  Serial.print(sensorValue3); // prints the value on the serial monitor
  Serial.print(",");
  Serial.print(accel.acceleration.x);
  Serial.print(",");
  Serial.print(accel.acceleration.y);
  Serial.print(",");
  Serial.print(accel.acceleration.z);
  Serial.print(",");
  Serial.print(gyro.gyro.x);
  Serial.print(",");
  Serial.print(gyro.gyro.y);
  Serial.print(",");
  Serial.print(gyro.gyro.z);
  Serial.println();
}
