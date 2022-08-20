#npk+gui
import time
import pymodbus
import serial
from pymodbus.pdu import ModbusRequest
from pymodbus.client.sync import ModbusSerialClient as ModbusClient #initialize a serial RTU client instance
from pymodbus.transaction import ModbusRtuFramer

import logging
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

import tkinter as tk
from tkinter import simpledialog
from tkinter import *
from tkinter import ttk

ROOT = tk.Tk()
ROOT.withdraw()




#count= the number of registers to read
#unit= the slave unit this request is targeting
#address= the starting address to read from

client= ModbusClient(method = "rtu", port="/dev/ttyUSB0",stopbits = 1, bytesize = 8, parity = 'E', baudrate= 9600)

#Connect to the serial modbus server
connection = client.connect()
#print (connection)

# address of the sensor data
N = '1e'
P = '1f'
K = '20'

global USER_INP


def read_sensor():
    global nitrogen
    N_data= client.read_holding_registers(int(N,16),1,unit= 1)
    nitrogen= N_data.registers[0]

    global phosphorus
    P_data= client.read_holding_registers(int(P,16),1,unit= 1)
    phosphorus= P_data.registers[0]
    
    global potassium
    K_data= client.read_holding_registers(int(K,16),1,unit= 1)
    potassium= K_data.registers[0]
    
    
def compare():
    global n_val,n_std
    global ph_val,ph_std
    global po_val,po_std
    global result
    
    if (USER_INP == "1"):
        print("Selected irrigated cotton")
        n_std =100
        ph_std=50
        po_std=50
        result ="Irrigated cotton"
        if (nitrogen<n_std): 
            n_val = n_std-nitrogen
        if (phosphorus<ph_std): 
            ph_val = ph_std-phosphorus
        if (potassium<po_std): 
            po_val = po_std-potassium
        else:
            n_val=0
            ph_val=0
            po_val=0
            
    elif (USER_INP == "2"):
        print("Selected rainfed cotton")
        n_std =80
        ph_std=40
        po_std=40
        result ="Rainfed cotton"
        if (nitrogen<n_std): #if (nitrogen<100 or phosphorus <50 or potassium<50):
            n_val = n_std-nitrogen
        if (phosphorus<ph_std): 
            ph_val = ph_std-phosphorus
        if (potassium<po_std): 
            po_val = po_std-potassium
        else:
            n_val=0
            ph_val=0
            po_val=0

    elif (USER_INP == "3"):
            print("Selected desi & hirsutum cotton")
            n_std =50
            ph_std=25
            po_std=25
            result ="Desi & Hirsutum cotton"
            if (nitrogen<n_std): #if (nitrogen<100 or phosphorus <50 or potassium<50):
                n_val = n_std-nitrogen
            if (phosphorus<ph_std): 
                ph_val = ph_std-phosphorus
            if (potassium<po_std): 
                po_val = po_std-potassium
            else:
                n_val=0
                ph_val=0
                po_val=0
                
    else:
        result ="Incorrect option"
        n_std =0
        ph_std=0
        po_std=0
        n_val=0
        ph_val=0
        po_val=0


    display()
        
        
            
def display():
    # Create an instance of tkinter frame
    win = Tk()

    # Set the size of the tkinter window
    win.geometry("700x140")
    win.title("Results for: " + result )
    

    # Create an object of Style widget
    style = ttk.Style()
    style.theme_use('clam')
    
    # Add a Treeview widget
    tree = ttk.Treeview(win, column=("Actual", "Standard", "Required"), show='headings', height=3)
    tree.column("# 1", anchor=CENTER)
    tree.heading("# 1", text="Actual")
    tree.column("# 2", anchor=CENTER)
    tree.heading("# 2", text="Standard")
    tree.column("# 3", anchor=CENTER)
    tree.heading("# 3", text="Required")
    
    
    # Insert the data in Treeview widget
    tree.insert('', 'end', text="1", values=(nitrogen, n_std, n_val))
    tree.insert('', 'end', text="1", values=(phosphorus, ph_std, ph_val))
    tree.insert('', 'end', text="1", values=(potassium, po_std, po_val))
    tree.pack()

    exit_button = Button(win, text="Exit", command=win.destroy,bg="white", fg="red")
    exit_button.place(x=320, y=110)
    win.mainloop()
    


read_sensor()
print("nitrogen: ",nitrogen,"mg/kg")
print("phosphorus: ",phosphorus,"mg/kg")
print("potassium: ",potassium,"mg/kg")
        
USER_INP = simpledialog.askstring(title="Comparision",prompt="Type your option: (1-2-3)")
#print(type(USER_INP))
compare()
        
time.sleep(2)
    
