# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:20:15 2023

@author: Rodrigo
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import datetime
import os

# LINHA EXEMPLO ANUNCIO - <h2 data-testid="ad-title" class="evg565y6 evg565y20 ooa-10p8u4x er34gjf0"><a href="https://www.standvirtual.com/anuncio/mercedes-benz-c-220-d-amg-line-ID8PqbkI.html" target="_self">Mercedes-Benz C 220 d AMG Line</a></h2>
# Verificar variavel 'texto', a classe costuma alterar
combined_df = pd.DataFrame()
pagina = 1
ultimapagina = 5
PriceFrom = [5000,15001,30001]
PriceTo = [15000,30000,45000]

for Price_From, Price_To in zip(PriceFrom, PriceTo):
    pagina = 1
    while pagina <= int(ultimapagina):
        url = 'https://www.standvirtual.com/carros/desde-2010?search%5Bfilter_float_price%3Afrom%5D='+str(Price_From)+'&search%5Bfilter_float_price%3Ato%5D='+str(Price_To)+'&page='+str(pagina)
        # url = "https://www.standvirtual.com/carros?page="+str(pagina)
        print(f'-- Página {pagina} -- De {Price_From} € a {Price_To} € -- SV --')
        response = requests.get(url)
        soup = BeautifulSoup(response.content,"html.parser")
        texto = soup.find_all("h1", class_="ev7e6t89 ooa-1xvnx1e er34gjf0") 
        count = 0
        for banners in texto:
            site = banners.find('a')['href']
            count += 1 
            response = requests.get(site)
        
            # Create a BeautifulSoup object from the HTML content
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find the "detalhes" section on the webpage
            detalhes_section = soup.find("h3", class_="e1iqsx45 ooa-vp6t6g")
            # detalhes_section = soup.find("h4", class_="offer-parameters__title")
            
            # Check if the "detalhes" section exists
            if detalhes_section:
                # Find the parent div of the "detalhes" section
                detalhes_div = detalhes_section.find_next("div", class_="offer-params")
            
                # Create a dictionary to store the extracted data
                detalhes_data = {}
            
                # Iterate over each "detalhes" item and extract the information
                for item in detalhes_div.find_all("li", class_="offer-params__item"):
                    key_element = item.find("span", class_="offer-params__label")
                    value_element = item.find("div", class_="offer-params__value")
            
                    if key_element and value_element:
                        key = key_element.text.strip()
                        value = value_element.text.strip()
                        detalhes_data[key] = value
                        
                 # Extract the "ad_price" separately
                ad_price_element = soup.find("span", class_="offer-price__number")
                if ad_price_element:
                    ad_price = ad_price_element.text.strip()
                    detalhes_data["Preco"] = ad_price
                    
                # Extract the "ad_id"
                ad_id_element = soup.find("span", id="ad_id")
                if ad_id_element:
                    ad_id = ad_id_element.text.strip()
                    detalhes_data["ID Anuncio"] = ad_id
                    
                 # Extract the "date"
                data_anuncio_element = soup.find("span", class_="offer-meta__value")
                if data_anuncio_element:
                    data_anuncio = data_anuncio_element.text.strip().split(", ")[1]
                    detalhes_data["Data Anuncio"] = data_anuncio

                # Extract the "Link"
                detalhes_data["Link"] = site

                # Create a DataFrame from the extracted data
                df = pd.DataFrame.from_dict(detalhes_data, orient="index", columns=["Detalhes"])
                
                # Append the individual DataFrame to the combined DataFrame
                combined_df = pd.concat([combined_df, df], axis=1)
                
                # Delay
                time.sleep(random.randint(2,6)) 
                print("Encontrado",count,"SV")
            else:
                print("Não foi encontrada a secção de 'detalhes'.")
    
        # Delay
        time.sleep(random.randint(1,3))   
        print("Página",pagina,", Done!")
        pagina = pagina + 1

combined_df = combined_df.transpose()
combined_df.reset_index(drop = True, inplace = True)
# combined_df.drop_duplicates(subset='ID Anuncio', inplace=True)
combined_df['Site'] = 'StandVirtual'
combined_df['Portugal'] = 'Yes'
combined_df['Data'] = current_date = datetime.datetime.now().strftime("%Y-%m-%d")

duplos = combined_df.duplicated().sum()  
print('Registos duplicados StandVirtual:',duplos)

# Get the current time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
file_name = f"StandVirtualGeral_{current_time}.csv"

# Construct the file name with the current time
wd = os.getcwd()
wd = os.path.abspath(os.path.join(wd, os.pardir))
path = os.chdir(wd)
path = os.path.join(wd,'Original Data','StandVirtual')
full_path = os.path.join(path, file_name)

# Save the DataFrame to CSV with the updated file name
combined_df.to_csv(full_path, index=False)
