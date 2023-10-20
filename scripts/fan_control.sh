#!/bin/bash

TEMP_PATH="/sys/devices/virtual/thermal/thermal_zone0/temp"
PWM_PATH="/sys/devices/pwm-fan/target_pwm"

while true; do
    TEMP=$(cat $TEMP_PATH)
    
    if [ $TEMP -lt 25000 ]; then
        PWM=0
    elif [ $TEMP -lt 50000 ]; then
        PWM=75
    elif [ $TEMP -lt 75000 ]; then
        PWM=150
    else
        PWM=255
    fi
    
    sh -c "echo $PWM > $PWM_PATH"
    cat $TEMP_PATH
    
    sleep 10
done
