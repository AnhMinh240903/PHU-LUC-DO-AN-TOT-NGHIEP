#!/home/anhminh/yolo/bin/python3


import time
from collections import deque
from typing import Dict, List

import Adafruit_ADS1x15
import RPi.GPIO as GPIO

__all__ = ["TactileSensor", "SENSOR_CONFIG"]




# I2C configuration for ADS1115
ADS_ADDRESS = 0x48
ADS_BUSNUM = 1
GAIN = 1
DATA_RATE = 475


ROW_CHS = [0, 1, 2]


SENSOR_CONFIG = [
    {
        "key": "L_OUT",
        "rows": 3,
        "cols": 2,
        "col_pins": [6, 26],
    },
    {
        "key": "L_IN",
        "rows": 3,
        "cols": 3,
        "col_pins": [4, 13, 27],
    },
    {
        "key": "R_IN",
        "rows": 3,
        "cols": 3,
        "col_pins": [24, 25, 5],
    },
    {
        "key": "R_OUT",
        "rows": 3,
        "cols": 2,
        "col_pins": [22, 23],
    },
]


SETTLE_TIME = 0.003  
OFF_DELAY = 0.0005  

# Calibration settings
CALIBRATE_FRAMES = 45  
RAW_THRESHOLD = 8000  


class TactileSensor:
   

    def __init__(self):
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        self.col_pins: List[int] = []
        for cfg in SENSOR_CONFIG:
            for p in cfg['col_pins']:
                if p not in self.col_pins:
                    self.col_pins.append(p)
                    GPIO.setup(p, GPIO.OUT)
                    GPIO.output(p, GPIO.LOW)
       
        self.adc = Adafruit_ADS1x15.ADS1115(address=ADS_ADDRESS, busnum=ADS_BUSNUM)
       
        self.baseline: Dict[str, List[List[float]]] = {}
        
        for cfg in SENSOR_CONFIG:
            key = cfg['key']
            rows, cols = cfg['rows'], cfg['cols']
            self.baseline[key] = [[0.0 for _ in range(cols)] for _ in range(rows)]

    
    def _all_cols_off(self):
        for p in self.col_pins:
            GPIO.output(p, GPIO.LOW)

    def _col_on(self, pin: int):
        GPIO.output(pin, GPIO.HIGH)

    def _col_off(self, pin: int):
        GPIO.output(pin, GPIO.LOW)

    def _read_adc(self, channel: int) -> int:
      
        return self.adc.read_adc(channel, gain=GAIN, data_rate=DATA_RATE)


    def calibrate(self, frames: int = CALIBRATE_FRAMES) -> None:
     
        
        accum: Dict[str, List[List[float]]] = {}
        count = 0
        for cfg in SENSOR_CONFIG:
            key = cfg['key']
            rows, cols = cfg['rows'], cfg['cols']
            accum[key] = [[0.0 for _ in range(cols)] for _ in range(rows)]
       
        for _ in range(frames):
            reading = self.read_raw()
            for cfg in SENSOR_CONFIG:
                key = cfg['key']
                rows, cols = cfg['rows'], cfg['cols']
                for r in range(rows):
                    for c in range(cols):
                        accum[key][r][c] += reading[key][r][c]
            count += 1
        
        for cfg in SENSOR_CONFIG:
            key = cfg['key']
            rows, cols = cfg['rows'], cfg['cols']
            for r in range(rows):
                for c in range(cols):
                    self.baseline[key][r][c] = accum[key][r][c] / float(count)

    def read_raw(self) -> Dict[str, List[List[float]]]:
        
        result: Dict[str, List[List[float]]] = {}
        
        self._all_cols_off()
        time.sleep(OFF_DELAY)
      
        for cfg in SENSOR_CONFIG:
            key = cfg['key']
            rows, cols = cfg['rows'], cfg['cols']
            col_pins = cfg['col_pins']
            mat = [[0.0 for _ in range(cols)] for _ in range(rows)]
            for c in range(cols):
                
                self._all_cols_off()
                time.sleep(OFF_DELAY)
                
                self._col_on(col_pins[c])
                time.sleep(SETTLE_TIME)
                
                for r in range(rows):
                    ch = ROW_CHS[r]
                    mat[r][c] = float(self._read_adc(ch))
               
                self._col_off(col_pins[c])
            result[key] = mat
        
        self._all_cols_off()
        return result

    def read_force(self) -> Dict[str, List[List[float]]]:
       
        raw = self.read_raw()
        forces: Dict[str, List[List[float]]] = {}
        for cfg in SENSOR_CONFIG:
            key = cfg['key']
            rows, cols = cfg['rows'], cfg['cols']
            mat = [[0.0 for _ in range(cols)] for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    
                    val = raw[key][r][c] - self.baseline[key][r][c]
                    # Apply threshold
                    if val < RAW_THRESHOLD:
                        val = 0.0
                    mat[r][c] = val
            forces[key] = mat
        return forces

    def cleanup(self):
        
        self._all_cols_off()
        GPIO.cleanup()


