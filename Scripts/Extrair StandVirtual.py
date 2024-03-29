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
ultimapagina = 4
PriceFrom = [5000,15001,30001]
PriceTo = [15000,30000,45000]

for Price_From, Price_To in zip(PriceFrom, PriceTo):
    pagina = 1
    while pagina <= int(ultimapagina):
        url = 'https://www.standvirtual.com/carros/desde-2014?search%5Bfilter_float_price%3Afrom%5D='+str(Price_From)+'&search%5Bfilter_float_price%3Ato%5D='+str(Price_To)+'&page='+str(pagina)
        # url = "https://www.standvirtual.com/carros?page="+str(pagina)
        print(f'-- Página {pagina} -- De {Price_From} € a {Price_To} € -- SV --')
        response = requests.get(url)
        soup = BeautifulSoup(response.content,"html.parser")
        texto = soup.find_all("article", class_="ooa-yca59n eszxync0") 
        count = 0
        for banners in texto:
            count += 1
            
            site = banners.find('a')['href']
            response = requests.get(site)
            soup = BeautifulSoup(response.content, "html.parser")
            
            detalhes_section = soup.find("div", class_="ooa-w4tajz e18eslyg0")
                        
            # Chech for DETAILS section
            if detalhes_section:

                details_divs = soup.find_all('div', class_='ooa-162vy3d e18eslyg3')

                detalhes_data = {}
                
                for div in details_divs:
                    title = div.find('p', class_='e18eslyg4 ooa-12b2ph5').text.strip()
                    
                    # Check if the value is in an <a> tag or a <p> tag
                    value_element = div.find('a', class_='e16lfxpc1 ooa-1ftbcn2') or div.find('p', class_='e16lfxpc0 ooa-1pe3502 er34gjf0')
                    value = value_element.text.strip() if value_element else None
                
                    detalhes_data[title] = value
                            ######################
                 
                 # PRICE
                ad_price_element = soup.find("h3", class_="offer-price__number eqdspoq4 ooa-o7wv9s er34gjf0")
                if ad_price_element:
                    ad_price = ad_price_element.text.strip()
                    detalhes_data["Preco"] = ad_price
                    
                # ID
                ad_id_element = soup.find('div', class_='ooa-1neiy54 edazosu6').find('p', class_='edazosu4 ooa-1afacld er34gjf0')
                if ad_id_element:
                    ad_id = ad_id_element.text.strip().split(": ")[1]
                    detalhes_data["ID Anuncio"] = ad_id
                    
                 # Banner DATE
                data_anuncio_element = soup.find('div', class_='ooa-1oivzan edazosu6').find('p', class_='edazosu4 ooa-1afacld er34gjf0')
                if data_anuncio_element:
                    data_anuncio = data_anuncio_element.text.strip().split("às")[0]
                    detalhes_data["Data Anuncio"] = data_anuncio

                # LINK
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
os.chdir(r'C:\Users\Rodrigo\Documents\GitHub\ImportedCars\Original Data\StandVirtual')
wd = os.getcwd()
full_path = os.path.join(wd, file_name)

combined_df.to_csv(full_path, index=False)
