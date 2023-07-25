import numpy as np
import pandas as pd

def calc_rx_power(d):
    # received power [mWh]
    return 1*10**((TX_POWER - (20*np.log10((4*np.pi*d*F)/C)))/10) * 1000

TX_POWER = 32
F=1e9
C=299792458
d=(((83-71)**2)+((93-89)**2)+100)**(1/2)

print(calc_rx_power(d))