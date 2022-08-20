import mysql.connector
global USER_INP
import cv2
import tensorflow as tf
import numpy as np
import os
import serial
import time
import pymodbus
from pymodbus.pdu import ModbusRequest
from pymodbus.client.sync import ModbusSerialClient as ModbusClient 
from pymodbus.transaction import ModbusRtuFramer
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
client= ModbusClient(method = "rtu", port="/dev/ttyUSB0",stopbits = 1, bytesize = 8, parity = 'E', baudrate= 9600)
connection = client.connect()
print("State : ",client.is_socket_open())
N = '1e'
P = '1f'
K = '20'


ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.reset_input_buffer()

vid = cv2.VideoCapture(0)

mysqldb = mysql.connector.connect(host="localhost", user="admin", password="admin", database="farmbot")

def convertTuple(tup):
    str = ''
    for item in tup:
        str = str + item
    return str	






while True:
    global result
    ret, frame = vid.read()
    cv2.imwrite('/var/www/html/output/1.jpg',frame)
    image_path = '/var/www/html/output/1.jpg'
    image=cv2.imread(image_path)
    name = os.path.split(image_path)[1]
    classes = ["Healthy", "Boll Rot", "Leafhopper Jassids", "Leaf Redenning","Rust of Cotton", "Wet Weather Blight", "White Flies"]
    interpreter = tf.lite.Interpreter(model_path = "model_cd_cnn.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = cv2.imread(image_path)
    output = image.copy()
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0
    input_shape = input_details[0]['shape']
    input_tensor = np.array(np.expand_dims(image, 0), dtype= "float32")
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    idx = pred.argsort()[-6:][::-1]
    result = dict(zip([classes[i] for i in idx],[round(pred[i]*100,2) for i in idx]))
    result=str(result)
    print(result)
    
    N_data= client.read_holding_registers(int(N,16),1,unit= 1)
    nsnr= N_data.registers[0]
    P_data= client.read_holding_registers(int(P,16),1,unit= 1)
    psnr= P_data.registers[0]
    K_data= client.read_holding_registers(int(K,16),1,unit= 1)
    print("K Data >> ",K_data)
    ksnr= K_data.registers[0]
    
    line = ser.readline().decode('utf-8').rstrip()
    print(">>>>",line)
    humsnr=line[0:5]
    print(humsnr)
    tempsnr=line[6:11]
    print(tempsnr)
    gassnr=line[12:18]
    print(gassnr)
    soilsnr=line[43:46]
    print(soilsnr)
    lightsnr=line[36:42]
    print(lightsnr)
    accsnr=line[24:35]
    print(accsnr)
    magnsnr=line[19:23]
    print(magnsnr)
    time.sleep(1)
    
    lft=" "
    mycursor=mysqldb.cursor()
    sql="INSERT INTO sensor_data(temperature_sensor,humidity_sensor,gas_sensor,light_sensor,n_value,p_value,k_value,soil_moisture,Accelerometer,Magnetometer,leaf_disease_predicted) VALUES(%s, %s, %s ,%s, %s, %s ,%s, %s, %s, %s, %s)"
    val=(tempsnr,humsnr,gassnr,lightsnr,nsnr,psnr,ksnr,soilsnr,accsnr,magnsnr,result)
    mycursor.execute(sql,val)
    mysqldb.commit()
    mycursor.execute("SELECT status FROM camera_status")
    myresult = mycursor.fetchall()
    for x in myresult:
        liste = str(list(x))
        liste=liste[1:2]
        liste=int(liste)
        print(liste)
        if liste==1:
            print("Result Value : ",result)
            sql="INSERT INTO sensor_data(temperature_sensor,humidity_sensor,gas_sensor,light_sensor,n_value,p_value,k_value,soil_moisture,Accelerometer,Magnetometer,leaf_disease_predicted) VALUES(%s, %s, %s ,%s, %s, %s ,%s, %s, %s, %s, %s)"
            val=(tempsnr,humsnr,gassnr,lightsnr,nsnr,psnr,ksnr,soilsnr,accsnr,magnsnr,result)
            mycursor.execute(sql,val)
            mysqldb.commit()
            sql = "UPDATE camera_status SET status = '0' WHERE id = '1'"
            mycursor.execute(sql)
            mysqldb.commit()

