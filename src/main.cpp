#include <Arduino.h>
#include "Arduino_GFX_Library.h"
#include "pin_config.h"
#include <Wire.h>
#include "HWCDC.h"
#include "font/Market_Deco12pt7b.h"
#include "font/Market_Deco24pt7b.h"
#include "font/Market_Deco32pt7b.h"
#include "ico/battery_ico.h"
#include "ico/falling_ico.h"
#include "time.h" // Required for ESP32 RTC
#include "SensorQMI8658.hpp"

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "../lib/model.h"

#define BAT_ADC_PIN 1 // GPIO1 connected to BAT_ADC
SensorQMI8658 qmi;

IMUdata acc;
IMUdata gyr;

Arduino_DataBus *bus = new Arduino_ESP32SPI(LCD_DC, LCD_CS, LCD_SCK, LCD_MOSI);
Arduino_GFX *gfx = new Arduino_ST7789(bus, LCD_RST, 0, true, LCD_WIDTH, LCD_HEIGHT, 0, 20, 0, 0);

// Dummy date and time initialization
struct tm timeInfo;
char *month[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JULY", "AUG", "SEP", "OCT", "NOV", "DEC"};

int fall_count = 0;
// Set initial date and time
void setRTC(int year, int month, int day, int hour, int minute)
{
  timeInfo.tm_year = year - 1900; // tm_year is years since 1900
  timeInfo.tm_mon = month - 1;    // tm_mon is 0-based
  timeInfo.tm_mday = day;
  timeInfo.tm_hour = hour;
  timeInfo.tm_min = minute;
  timeInfo.tm_sec = 0;

  time_t t = mktime(&timeInfo);
  struct timeval now = {.tv_sec = t};
  settimeofday(&now, NULL);
}

int mapVoltageToPercentage(float voltage)
{
  if (voltage <= 2.5)
    return 0; // Minimum voltage corresponds to 0%
  if (voltage >= 3.7)
    return 100;                                          // Maximum voltage corresponds to 100%
  return (int)((voltage - 2.5) * (100.0 / (3.7 - 2.5))); // Map 2.5V - 3.7V to 0% - 100%
}

int readBatteryVoltage()
{
  int adcValue = analogRead(BAT_ADC_PIN);    // Read the ADC value (0-4095 for 12-bit ADC)
  float voltage = (adcValue / 4095.0) * 3.3; // Convert ADC value to voltage (3.3V reference)
  voltage = voltage * ((200000.0 + 100000.0) / 100000.0);
  mapVoltageToPercentage(voltage);
  return mapVoltageToPercentage(voltage); // Adjust for the voltage divider (R3 and R7)
}

// Display the current date and time
void displayDateTime()
{
  gfx->fillScreen(BLACK);
  gfx->setFont(&Market_Deco32pt7b);
  gfx->setTextColor(WHITE);
  gfx->setCursor(42, 120);
  gfx->setTextSize(1);

  if (timeInfo.tm_hour < 10)
    gfx->print("0");
  gfx->print(timeInfo.tm_hour);
  gfx->print(" ");
  if (timeInfo.tm_min < 10)
    gfx->print("0");
  gfx->println(timeInfo.tm_min);

  gfx->setFont(&Market_Deco12pt7b);
  gfx->setTextSize(1);
  gfx->setCursor(50, 180);
  gfx->print(timeInfo.tm_mday);
  gfx->printf(" %s ", month[timeInfo.tm_mon]); // Replace with month if dynamic names needed
  gfx->println(1900 + timeInfo.tm_year);
  gfx->setCursor(108, 215);
  gfx->print(readBatteryVoltage());
  gfx->println("%");
  gfx->drawBitmap(85, 200, battery_icon, 16, 16, WHITE);
  gfx->setTextColor(YELLOW);
  gfx->drawBitmap(85, 240, falling_icon, 24, 24, YELLOW);
  gfx->setCursor(120, 258);
  gfx->println(fall_count);
}

// Task for updating time every minute
void timeUpdateTask(void *parameter)
{
  while (1)
  {
    // Retrieve current time from RTC
    if (getLocalTime(&timeInfo))
    {
      displayDateTime();
    }
    vTaskDelay(60000 / portTICK_PERIOD_MS); // 1 minute delay
  }
}

/************************************************************************************************* */
#pragma region TFlite
const float accelerationThreshold = 2.3; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char *GESTURES[] = {
    "normal",
    "nearfall"};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

/************************************************************************************************* */

void setup()
{
  USBSerial.begin(115200);
  // while (!USBSerial);

  USBSerial.println("Arduino_GFX Hello World example with RTC");
  analogReadResolution(12);
  // Init Display
  if (!gfx->begin())
  {
    USBSerial.println("gfx->begin() failed!");
  }
  gfx->fillScreen(BLACK);

  pinMode(LCD_BL, OUTPUT);
  analogWrite(LCD_BL, 100);

  // Set initial date and time (YYYY, MM, DD, HH, MM)
  setRTC(2024, 10, 31, 6, 45); // Adjust to the current date and time

  // Initial display
  displayDateTime();

  // Start task for time updates
  xTaskCreatePinnedToCore(timeUpdateTask, "Time Update Task", 2048, NULL, 1, NULL, 1);

  if (!qmi.begin(Wire, QMI8658_L_SLAVE_ADDRESS, IIC_SDA, IIC_SCL))
  {
    USBSerial.println("Failed to find QMI8658 - check your wiring!");
    while (1)
    {
      delay(1000);
    }
  }
  /* Get chip id*/
  USBSerial.print("Device ID:");
  USBSerial.println(qmi.getChipID(), HEX);

  qmi.configAccelerometer(
      SensorQMI8658::ACC_RANGE_4G,
      SensorQMI8658::ACC_ODR_1000Hz,
      SensorQMI8658::LPF_MODE_0,
      true);

  qmi.configGyroscope(
      SensorQMI8658::GYR_RANGE_64DPS,
      SensorQMI8658::GYR_ODR_896_8Hz,
      SensorQMI8658::LPF_MODE_3,
      true);

  // In 6DOF mode (accelerometer and gyroscope are both enabled),
  // the output data rate is derived from the nature frequency of gyroscope
  qmi.enableGyroscope();
  qmi.enableAccelerometer();

  // Print register configuration information
  qmi.dumpCtrlRegister();

  USBSerial.println("Read data now...");

  pinMode(36, INPUT_PULLUP);

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {
    USBSerial.println("Model schema mismatch!");
    while (1)
      ;
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop()
{
  float aX, aY, aZ, gX, gY, gZ;
  // Main loop can be used for other tasks or remain empty
  // if (qmi.getDataReady())
  // {
  //   qmi.getAccelerometer(acc.x, acc.y, acc.z);
  //   qmi.getGyroscope(gyr.x, gyr.y, gyr.z);
  // }
  if (digitalRead(36) == LOW)
  {
    gfx->fillRect(85, 195, 80, 40, BLACK);
    gfx->setCursor(95, 215);
    gfx->setTextColor(RED);
    gfx->println("SOS");
    delay(3000);
    gfx->setTextColor(WHITE);
    gfx->fillRect(85, 195, 80, 40, BLACK);
    gfx->setCursor(108, 215);
    gfx->print(readBatteryVoltage());
    gfx->println("%");
    gfx->drawBitmap(85, 200, battery_icon, 16, 16, WHITE);
  }
  // wait for significant motion
  while (samplesRead == numSamples)
  {
    if (qmi.getDataReady())
    {
      // read the acceleration data
      qmi.getAccelerometer(aX, aY, aZ);

      // sum up the absolutes
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold)
      {
        // reset the sample read count
        samplesRead = 0;
        break;
      }
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples)
  {
    // check if new acceleration AND gyroscope data is available
    if (qmi.getDataReady())
    {
      // read the acceleration and gyroscope data
      // IMU.readAcceleration(aX, aY, aZ);
      // IMU.readGyroscope(gX, gY, gZ);
      qmi.getAccelerometer(aX, aY, aZ);
      qmi.getGyroscope(gX, gY, gZ);

      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples)
      {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk)
        {
          USBSerial.println("Invoke failed!");
          while (1)
            ;
          return;
        }

        // Loop through the output tensor values from the model
        for (int i = 0; i < NUM_GESTURES; i++)
        {
          USBSerial.print(GESTURES[i]);
          USBSerial.print(": ");
          USBSerial.println(tflOutputTensor->data.f[i], 6);
          if(i == 1){
            if(tflOutputTensor->data.f[i] > 0.7){
              fall_count++;
              gfx->fillRect(115, 250, 80, 40, BLACK);
              gfx->setCursor(120, 258);
              gfx->println(fall_count);
            }
          }
        }
        USBSerial.println();
      }
    }
  }
  delay(5);
}
