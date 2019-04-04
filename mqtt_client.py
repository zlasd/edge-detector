import base64
import time
import json
from datetime import datetime

import paho.mqtt.client as mqtt

SERIAL_NUMBER = '00000000060ce379'
DEVICE_PASSWD = 'rasppi'
DEVICE_ADDR = '教学楼'

MQTT_SERVER = ''
ENCODING = 'utf-8'

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

def on_publish(client, userdata, mid):
    print("Published message #" + str(mid))

    
client = mqtt.Client("publisher")
client.on_connect = on_connect
client.on_publish = on_publish
client.connect(MQTT_SERVER, 1883, 60)
client.loop_start()

def sendMQTT(imlist, confidence):
    from my_utils import createGif
    fname = str(datetime.now().timestamp()) + '.gif'
    createGif(fname, imlist)
    
    with open(fname, 'rb') as img:
        byte_content = img.read()
    base64_bytes = base64.b64encode(byte_content)
    base64_string = base64_bytes.decode(ENCODING)
    
    raw_data = {}
    raw_data["serial-number"] = SERIAL_NUMBER
    raw_data["passwd"] = DEVICE_PASSWD
    raw_data["image_base64_string"] = base64_string
    raw_data["personNo"] = 2
    raw_data["confidence"] = confidence
    
    json_data = json.dumps(raw_data)
    client.publish("/device/alert", json_data, 1)
    print("loop")
    
    
if __name__ == '__main__':
    raw_data = {}
    raw_data["serial-number"] = SERIAL_NUMBER
    raw_data["passwd"] = DEVICE_PASSWD
    raw_data["type"] = "Raspberry Pi"
    raw_data["address"] = DEVICE_ADDR
    
    json_data = json.dumps(raw_data)
    client.publish("/device/add", json_data, 1)
    time.sleep(3)
    client.disconnect()