#include <AccelStepper.h>

#define dirPin 9
#define stepPin 6
#define stepsPerRevolution 3200
AccelStepper stepper = AccelStepper(1, stepPin, dirPin);

void setup()
{
  Serial.begin(9600);

  stepper.setMaxSpeed(200);
  stepper.setAcceleration(400);
}

void loop()
{
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    Serial.println(command);
    executeCommand(command);
  }

  stepper.run();
}

void executeCommand(String command)
{
  if (command.startsWith("degree")){
    float degrees = command.substring(6).toFloat();
    moveDegrees(degrees);
  }else if (command.startsWith("step")){
    String steps = command.substring(4);
    Serial.println(steps);
    moveSteps(steps.toInt());
  }else if (command.equals("reset")){ 
    resetMotor();
  }
}

void moveDegrees(float degrees)
{
  Serial.print("move ");  Serial.print(degrees);  Serial.println (" degrees");
  float targetPosition = stepper.currentPosition() + degrees * (stepsPerRevolution / 360.0);
  stepper.moveTo(targetPosition);
}

void moveSteps(int steps)
{
  Serial.print("move ");  Serial.print(steps);  Serial.println (" steps");
  stepper.move(steps);
}

void resetMotor()
{
  Serial.println ("reset motor");
  stepper.moveTo(0);
}
