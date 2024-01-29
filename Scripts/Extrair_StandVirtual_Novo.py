# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:16:19 2023

@author: Rodrigo
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import datetime
# import os

combined_df = pd.DataFrame()
pagina = 1
ultimapagina = 2
PriceFrom = [5000]#,15001,30001]
PriceTo = [15000]#,30000,45000]

for Price_From, Price_To in zip(PriceFrom, PriceTo):
    while pagina <= int(ultimapagina):
        url = 'https://www.standvirtual.com/carros/desde-2010?search%5Bfilter_float_price%3Afrom%5D='+str(Price_From)+'&search%5Bfilter_float_price%3Ato%5D='+str(Price_To)+'&page='+str(pagina)
            
        print(f'-- Página {pagina} -- De {Price_From} € a {Price_To} € -- SV --')

        response = requests.get(url)
        soup = BeautifulSoup(response.content,"html.parser")
        texto = soup.find_all("h1", class_="e1oqyyyi9 ooa-1ed90th er34gjf0") 
        count = 0
        
        for banners in texto:
            car_info_list = []
            
            count += 1 
            if count > 3:
                break

            site = banners.find('a')['href']
            response = requests.get(site)
            soup = BeautifulSoup(response.content,"html.parser")
            items = soup.find_all('div', class_="ooa-162vy3d e18eslyg3")
            for item in items:
                parts = item.get_text(separator=':').split(':')
                
                for i in range(0,len(parts),2):
                     car_info = {parts[i]:parts[i+1]}   
                     car_info_list.append(car_info)
                     df = pd.DataFrame(car_info_list)
                     df = df.transpose()
                        
                result_df = pd.concat([df[col] for col in df.columns]).dropna()
                combined_df = pd.concat([combined_df, result_df], axis=1)
            
            # Delay
            time.sleep(random.randint(2,6)) 
            print("Encontrado",count,"SV")
   
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
