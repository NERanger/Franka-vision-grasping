import Jetson.GPIO as GPIO

valve_pin = 16
GPIO.setmode(GPIO.BOARD)
GPIO.setup(valve_pin, GPIO.OUT)

def gripper_open():
    GPIO.output(valve_pin, GPIO.HIGH)

def gripper_close():
    GPIO.output(valve_pin, GPIO.LOW)

if __name__ == '__main__':
    import time

    while(True):
        gripper_open()
        time.sleep(2)
        gripper_close()
        time.sleep(2)