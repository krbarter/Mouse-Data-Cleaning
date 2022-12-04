import os
import cv2
import fileinput
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

name = input("Enter a file name: ")

# section one intial processing
df = pd.read_csv(name + ".csv")
df.rename(columns = {' Scan No.':'Scan No.', ' Depth':'Depth', ' Layer':'Layer'}, inplace = True)
cols = ['Index', 'Scan No.', 'Depth', 'Layer']
df = df[cols]
df['Depth'] = df['Depth'].replace([-1], 0.0)

# section two - formating
df_format = df.groupby(["Scan No.", "Layer"]).agg({'Depth':"mean"}).reset_index()
df_format = df_format.pivot(index = "Scan No.", columns = "Layer", values = "Depth")

# new df with average layer insead of avg position
layer = pd.DataFrame()
layer["NFL/GLC"] = df_format["ORNFL"] - df_format["IRNFL"]
layer["IPL"]     = df_format["IPL"]   - df_format["ORNFL"]
layer["INL"]     = df_format["INL"]   - df_format["IPL"]
layer["OPL"]     = df_format["OPL"]   - df_format["INL"]
layer["ONL-IS"]  = df_format["IS"]    - df_format["OPL"]
layer["OS"]      = df_format["ETPR"]  - df_format["IS"]
layer["RPE"]     = df_format["RPE"]   - df_format["ETPR"]
layer["TOTAL"]   = df_format["RPE"]   - df_format["ORNFL"]
layer["OVERALL"] = (layer["NFL/GLC"] + layer["IPL"] + layer["INL"] + layer["OPL"] + layer["ONL-IS"] + layer["OS"] + layer["RPE"]) / 7

# all data
layer.to_csv(name + "Mouse_OUT_FULL.csv", encoding='utf-8')
mask = layer["OVERALL"] != 0
layer[mask]
layer.to_csv(name + "Mouse_OUT.csv", encoding='utf-8')

#creating the image sets to be processed
def openDirectory(directory_name):
    imgs = []
    if os.path.isdir(directory_name) == True:
        for x in os.listdir(directory_name):
            if x.endswith(".TIFF"):
                imgs.append(os.path.join(directory_name, x))
    else:
        return "Input is not a valid drectory"
    return imgs

imgs = openDirectory(name)
ETPR  = np.zeros((100,1000), dtype = "int32")
INL   = np.zeros((100,1000), dtype = "int32")
IPL   = np.zeros((100,1000), dtype = "int32")
IRNFL = np.zeros((100,1000), dtype = "int32")
IS    = np.zeros((100,1000), dtype = "int32")
OPL   = np.zeros((100,1000), dtype = "int32")
ORNFL = np.zeros((100,1000), dtype = "int32")
RPE   = np.zeros((100,1000), dtype = "int32")

mask = df["Depth"] != 0
df = df[mask]

def formats(row):
    scale = row["Index"] - 1
    while scale > 1000:
        scale = scale - 1000
        
    if row["Layer"] == "ETPR":
        ETPR[row["Scan No."] - 1][scale] = int(row["Depth"])
    elif row["Layer"] == "INL":
        INL[row["Scan No."] - 1][scale] = int(row["Depth"])
    elif row["Layer"] == "IPL":
        IPL[row["Scan No."] - 1][scale] = int(row["Depth"])
    elif row["Layer"] == "IRNFL":
        IRNFL[row["Scan No."] - 1][scale] = int(row["Depth"])  
    elif row["Layer"] == "IS":
        IS[row["Scan No."] - 1][scale] = int(row["Depth"])
    elif row["Layer"] == "OPL":
        OPL[row["Scan No."] - 1][scale] = int(row["Depth"]) 
    elif row["Layer"] == "ORNFL":
        ORNFL[row["Scan No."] - 1][scale] = int(row["Depth"])
    elif row["Layer"] == "RPE":
        RPE[row["Scan No."] - 1][scale] = int(row["Depth"]) 
    return row

df.apply(formats, axis = 1)


save_location = os.path.join(os.getcwd(), "OutImages")

# making diretories
if os.path.isdir(save_location) == False:
    os.mkdir(save_location)
if os.path.isdir(os.path.join(save_location, name)) == False:
    os.mkdir(os.path.join(save_location, name))

#each layer its colour
for x in range(0, len(imgs)):
    image = cv2.imread(imgs[x])
    for y in range(0, 1000):
        if   IRNFL[x][y] != 0: image[IRNFL[x][y]][y]  = [255,0,0]
        if   ORNFL[x][y] != 0: image[ORNFL[x][y]][y]  = [0,0,255]
        if   IPL[x][y]   != 0: image[IPL[x][y]][y]    = [0,255,0]
        if   INL[x][y]   != 0: image[INL[x][y]][y]    = [255,105,180]
        if   OPL[x][y]   != 0: image[OPL[x][y]][y]    = [255,128,0]
        if   IS[x][y]    != 0: image[IS[x][y]][y]     = [0,255,255]
        if   ETPR[x][y]  != 0: image[ETPR[x][y]][y]   = [0,0, 255]
        if   RPE[x][y]   != 0: image[RPE[x][y]][y]    = [255,255,0]
    mpimg.imsave(os.path.join(save_location, imgs[x]), image)

# heatmap intro - getting the heatmap total value matrix working
total = np.zeros((100, 1000), dtype = "int32")
for x in range(0, 100):
    for y in range(0,1000):
        total[x][y] = int(RPE[x][y]) - int(ORNFL[x][y])

new = []
for x in range(0, 100):
    for y in range(0, 10):
        new.append(total[x])

total = np.array(new)

for x in range(0, 1000):
    for y in range(0, 1000):
        m = 100
        if total[x][y] - m < 0:
            total[x][y] = 0
        else: 
            total[x][y] = total[x][y] - m

blank_image = np.zeros((1030,1000,3), np.uint8)
color_gradient = [[0,100,250], [0,95,235], [0,90,255], [0,85,212], [0,80,199], [0,75,186],
                        [0,70,175],  [0,65,162], [0,60,151], [0,55,138], [0,50,125],                       #Blue

                        [9,255,0], [8,245,0], [8,235,0], [7,225,0], [7,215,0], [6,205,0],
                        [6,195,0], [5,185,0], [4,175,0], [4,165,0], [3,155,0],                               #Green

                        [255,255,0], [250,250,0], [245,245,0], [240,240,0], [235,235,0],
                        [230,230,0], [225,225,0], [220,220,0], [215,215,0], [210,210,0], [205,205,0],        #Yellow

                        [255,119,0], [245,117,0], [235,110,0], [225,105,0], [215,100,0], [205,96,0],
                        [195,91,0], [185,86,0], [175,83,0], [165,78,0], [155,72,0],                          #Orange

                        [255,0,0], [240,0,0], [225,0,0], [210,0,0], [195,0,0], [180,0,0],
                        [165,0,0], [150,0,0], [135,0,0], [120,0,0], [105,0,0]]                               #Red

for x in range(0,1000):
    for y in range(0,1000):
        if total[x][y] == 0:
             blank_image[x][y] = [0,0,0]
        elif total[x][y] >= len(color_gradient):
            blank_image[x][y] = [105,0,0]
        else:
            blank_image[x][y] = color_gradient[total[x][y]]
            
# blank lines
# the colour key at the bottom of the image
color_key = [x for x in color_gradient for i in range(10)]
while len(color_key) < 1000:
    color_key.insert(0, [0,0,0])
    if len(color_key) < 1000:
        color_key.append([0,0,0])
        
for x in range(1020, 1030):
    for y in range(0,len(color_key)):
        blank_image[x][y] = color_key[y]

maxval = str(np.max(total))
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(blank_image, "Min" ,(230, 1010), font, 1,(255,255,255),1,cv2.LINE_AA)
cv2.putText(blank_image, str(m),(300, 1010), font, 1,(255,255,255),1,cv2.LINE_AA)
cv2.putText(blank_image, "Max" ,(660, 1010), font, 1,(255,255,255),1,cv2.LINE_AA)
cv2.putText(blank_image, maxval,(730, 1010), font, 1,(255,255,255),1,cv2.LINE_AA)
mpimg.imsave(save_location + str(os.sep) + str(m) + "Heatmap" + ".TIFF", blank_image)