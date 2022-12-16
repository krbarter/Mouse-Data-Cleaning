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
heatmap_options = "Original"
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

# original
Original = [[0,0,0],[0,100,250], [0,95,235], [0,90,255], [0,85,212], [0,80,199], [0,75,186], [0,70,175],  [0,65,162], 
[0,60,151], [0,55,138], [0,50,125],[9,255,0], [8,245,0], [8,235,0], [7,225,0], [7,215,0], [6,205,0],[6,195,0], [5,185,0], 
[4,175,0], [4,165,0], [3,155,0],[255,255,0], [250,250,0], [245,245,0], [240,240,0], [235,235,0],[230,230,0], [225,225,0], 
[220,220,0], [215,215,0], [210,210,0], [205,205,0],[255,119,0], [245,117,0], [235,110,0], [225,105,0], [215,100,0], [205,96,0],
[195,91,0], [185,86,0], [175,83,0], [165,78,0], [155,72,0], [255,0,0], [240,0,0], [225,0,0], [210,0,0], [195,0,0], [180,0,0],
[165,0,0], [150,0,0], [135,0,0], [120,0,0], [105,0,0]]

# Colour Blind - Viridis
Viridis = [[0,0,0],[253,231,37], [244,230,30], [231,228,25], [218,227,25], [208,225,28], [194,223,35],
[181,222,43], [168,219,52], [157,217,59], [144,215,67], [132,212,75], [119,209,83], [110,206,88], [99, 203, 95],
[88,199,101], [78,195,107], [70,192,111], [61,188,116], [53,183,121], [46,179,124], [41,175,127], [37,171,130],
[33,166,133], [31,161,135], [31,158,137], [31,153,138], [31,148,140], [33,145,140], [34,140,141], [36,135,142],
[38,130,142], [39,127,142], [41,122,142], [43,117,142], [45,113,142], [46,109,142], [49,104,142], [51,99,141],
[53,94,141], [55,90,140], [58,84,140], [60,79,138], [62,73,137], [64,69,136], [66,63,133], [68,57,131], [70,51,127],
[71,46,124], [72,40,120], [72,33,115], [72,27,109], [72,22,104], [71,14,97], [70,7,90], [68,1,84]]

# Colur Blind #2 - Plasma
Plasma = [[0,0,0], [253,231,37], [244,230,30], [231,228,25], [218,227,25], [208,225,28], [194,223,35],
[181,222,43], [168,219,52], [157,217,59], [144,215,67], [132,212,75], [119,209,83], [110,206,88], [99,203,95], [88,199,101],
[78,195,107], [70,192,111], [61,188,116], [53,183,121], [46,179,124], [41,175,127], [37,171,130], [33,166,133], [31,161,135],
[31,158,137], [31,153,138], [31,148,140], [33,145,140], [34,140,141], [36,135,142], [38,130,142], [39,127,142], [41,122,142],
[43,117,142], [45,113,142], [46,109,142], [49,104,142], [51,99,141], [53,94,141], [55,90,140], [58,84,140], [60,79,138], [62,73,137],
[64,69,136], [66,63,133], [68,57,131], [70,51,127], [71,46,124], [72,40,120], [72,33,115], [72,27,109], [72,22,104], [71,14,97],
[70,7,90], [68,1,84]]

# Colur Blind #2 - Inferno
Inferno = [[0,0,0], [252,255,164], [246,250,150], [242,244,130], [241,236,109], [243,229,93], [245,219,76], [247,209,61],
[249,199,47], [250,192,38], [251,182,26], [252,172,17], [252,163,9], [251,155,6], [250,146,7], [248,137,12], [246,128,19], [244,121,24],
[241,113,31], [237,105,37], [233,97,43], [229,92,48], [224,85,54], [218,78,60], [212,72,66], [207,68,70], [200,63,75], [193,58,80], [188,55,84], 
[180,51,89], [173,48,93], [165,44,96], [159,42,99], [151,39,102], [143,36,105], [135,33,107], [128,31,108], [120,28,109], [113,25,110], [105,22,110],
[98,20,110], [90,17,110], [82,14,109], [74,12,107], [68,10,104], [59,9,100], [50,10,94], [41,11,85], [35,12,76], [27,12,65], [20,11,52], [13,8,41], 
[9,6,31], [4,3,20], [2,1,10], [0,0,4]]

# Colur Blind #2 - Magma
Magma = [[0,0,0], [252,253,191], [252,246,184], [252,236,174], [253,227,165], [253,220,158], [254,211,149], [254,202,141], [254,193,133], 
[254,185,127], [254,176,120], [254,167,114], [254,157,108], [253,150,104], [252,140,99], [251,131,95], [249,121,93], [247,114,92], [244,105,92], [241,96,93], 
[236,88,96], [232,83,98], [226,77,102], [219,71,106], [211,67,110], [205,64,113], [197,60,116], [189,57,119], [183,55,121], [174,52,123], [166,49,125], 
[158,47,127], [152,45,128], [144,42,129], [136,39,129], [128,37,130], [121,34,130], [114,31,129], [106,28,129], [98,25,128], [92,22,127], [84,19,125],
[76,17,122], [68,15,118], [61,15,113], [52,16,105], [44,17,95], [36,18,83], [30,17,73], [24,15,61], [18,13,49], [12,9,38], [8,7,30], [4,4,20], [2,1,9], [0,0,4]]

print(heatmap_options)
color_gradient = []
if heatmap_options == "Original" or heatmap_options == "original":
    color_gradient = Original
if heatmap_options == "Viridis"  or heatmap_options == "viridis":
    color_gradient = Viridis
if heatmap_options == "Plasma"   or heatmap_options == "plasma":
    color_gradient = Plasma
if heatmap_options == "Inferno"  or heatmap_options == "inferno":
    color_gradient = Inferno
if heatmap_options == "Magma"    or heatmap_options == "magma":
    color_gradient = Magma

for x in range(0,1000):
    for y in range(0,1000):
        if total[x][y] == 0:
             blank_image[x][y] = [0,0,0]
        elif total[x][y] >= len(color_gradient):
            blank_image[x][y] = color_gradient[-1]
        else:
            blank_image[x][y] = color_gradient[total[x][y]]

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