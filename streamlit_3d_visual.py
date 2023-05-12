import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

filename = 'supplychain_data_clean.csv'
df = pd.read_csv(filename)
data = df[['S_D1','s_D1','S_D2','s_D2','S_R1','s_R1','S_R2','s_R2','avg_net_profit']].values
data = np.array(data)

#finding unique values
SD = np.unique(data[:,0])
sD = np.unique(data[:,1])
SR = np.unique(data[:,4])
sR = np.unique(data[:,5])

stepSD = SD[1] - SD[0]
stepsD = sD[1] - sD[0]
SD_val = st.slider(label="S_D",min_value=int(min(SD)),max_value=int(max(SD)),step=int(stepSD))
sD_val = st.slider(label="s_D",min_value=int(min(sD)),max_value=int(max(sD)),step=int(stepsD))


print(f"SD = {SD}, sD = {sD}")
data_sec = data[data[:,0]==SD_val]
data_sec = data_sec[data_sec[:,1]==sD_val]
print(f"Slice at S_D = {SD_val}, s_D = {sD_val}, section size = {data_sec.shape}")

if(data_sec.shape[0]>3):
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_trisurf(data_sec[:,4],data_sec[:,5], data_sec[:,8],color='c',alpha=0.85,antialiased=True)
    plt.xlabel("S_R")
    plt.ylabel("s_R")
    st.pyplot(fig)
else:
    st.header("Not Enough points available!")