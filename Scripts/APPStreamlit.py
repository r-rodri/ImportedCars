# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 21:54:54 2023

@author: Rodrigo
"""

import streamlit as st
import subprocess

# Function to run the selected Python script
def run_script(script_name):
    st.write(f"Running {script_name}...")
    process = subprocess.Popen(["python", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    st.write("Output:")
    st.write(out.decode("utf-8"))
    st.write("Error:")
    st.write(err.decode("utf-8"))


def all_scripts():
    all_scripts = ['Cleaning AutoScout24.py','Cleaning StandVirtual.py','MergedFiles','RAW_Analise Diferencas Preco.py']
    for scripts in all_scripts:
        run_script(scripts)

# Streamlit app
st.title('Script Runner App')

# Button to run Script 1
if st.button('Run Script'):
    all_scripts()
    
# # Button to run Script 2
# if st.button('Run Script 2'):
#     run_script('script2.py')

# # Button to run Script 3
# if st.button('Run Script 3'):
#     run_script('script3.py')

# # Button to run Script 4
# if st.button('Run Script 4'):
#     run_script('script4.py')
