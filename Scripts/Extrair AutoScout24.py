# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:06:35 2023

@author: Rodrigo
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import datetime
import os
# import logging

# logging.basicConfig(
#     filename='TesteLog.log',
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     level=logging.Info,
# )

# Função para colocar em branco alguma variavel (ex.: marca, modelo, versao) nao encontrada no html. 
# E assim evitar a paragem abrupta do scrip. 
def good_soup(soup, divisor, classe= None):
    divisor_encontrado = soup.find(divisor, class_= classe)
    if divisor_encontrado:
        return divisor_encontrado.text.strip()
    else:
        return ''

car_listings = []
pagina = 1
ultimapagina = 3
PriceFrom = [5000,15001,30001]
PriceTo = [15000,30000,45000]

# Loop para percorrer os diferentes preços no site
for Price_From, Price_To in zip(PriceFrom, PriceTo):
    pagina = 1
    while pagina <= int(ultimapagina):
       
        # Usar para vendedores colectivos = empresa
        # url = 'https://www.autoscout24.com/lst?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&fregfrom=2014&kmto=150000&page='+str(pagina)+'&powertype=kw&pricefrom='+str(Price_From)+'&priceto='+str(Price_To)+'&search_id=1size64tibf&sort=standard&source=listpage_pagination&ustate=N%2CU'
        
        # Usar para vendedores privados
        # url = 'https://www.autoscout24.com/lst?atype=C&custtype=P&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&fregfrom=2010&page='+str(pagina)+'&powertype=kw&pricefrom='+str(Price_From)+'&priceto='+str(Price_To)+'&search_id=zz6yqqop1w&sort=standard&source=listpage_pagination&ustate=N%2CU'
       
        # BMW+VW+Ford
        # url = 'https://www.autoscout24.com/lst/bmw?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&desc=0&fregfrom=2010&fregto=2022&kmto=150000&mmmv=29%7C%7C%7C%2C74%7C%7C%7C&page='+str(pagina)+'&powertype=kw&pricefrom='+str(Price_From)+'&priceto='+str(Price_To)+'&search_id=1giwjlvcsmx&sort=standard&source=listpage_pagination&ustate=N%2CU' 
       
        # Electricos+Hibridos + >2016
        url = 'https://www.autoscout24.com/lst?atype=C&cy=D%2CA%2CB%2CE%2CF%2CI%2CL%2CNL&damaged_listing=exclude&desc=0&fregfrom=2016&fuel=E%2C2%2C3&kmto=80000&page='+str(pagina)+'&powertype=kw&pricefrom='+str(Price_From)+'&priceto='+str(Price_To)+'&search_id=walm73yy7s&sort=standard&source=listpage_pagination&ustate=N%2CU'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        count = 0 
        texto = soup.find_all('article', class_="cldt-summary-full-item listing-impressions-tracking list-page-item ListItem_article__qyYw7")
        print(f'-- Página {pagina} -- De {Price_From} € a {Price_To} € -- AS --')
        for banners in texto:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            site = banners.find('a')['href']
            fullsite = 'https://www.autoscout24.com' + site
            count += 1 
            response = requests.get(fullsite)
            soup = BeautifulSoup(response.content, "html.parser")
            
            id_ = site.split('-')
            new_id = ''.join(id_[-5:])
            
            make = good_soup(soup,'span', classe="StageTitle_boldClassifiedInfo__sQb0l")
            model = good_soup(soup,'span', classe="StageTitle_model__EbfjC StageTitle_boldClassifiedInfo__sQb0l")
            modelVersion = good_soup(soup,'div', classe="StageTitle_modelVersion__Yof2Z")
            mileage = good_soup(soup,'div', classe="VehicleOverview_itemText__AI4dA")
            
            
            overview_items = soup.find_all('div', class_="VehicleOverview_itemContainer__XSLWi")
            
            mileage = gearbox = first_registration = fuel_type = power = seller = price = ''

            try:
                mileage = overview_items[0].find('div', class_="VehicleOverview_itemText__AI4dA").text.strip()
            except IndexError:
                pass
            
            try:
                gearbox = overview_items[1].find('div', class_="VehicleOverview_itemText__AI4dA").text.strip()
            except IndexError:
                pass
            
            try:
                first_registration = overview_items[2].find('div', class_="VehicleOverview_itemText__AI4dA").text.strip()
            except IndexError:
                pass
            
            try:
                fuel_type = overview_items[3].find('div', class_="VehicleOverview_itemText__AI4dA").text.strip()
            except IndexError:
                pass
            
            try:
                power = overview_items[4].find('div', class_="VehicleOverview_itemText__AI4dA").text.strip()
            except IndexError:
                pass
            
            try:
                seller = overview_items[5].find('div', class_="VehicleOverview_itemText__AI4dA").text.strip()
            except IndexError:
                pass
            
            try:
                price = soup.find('span', class_="PriceInfo_price__XU0aF").text.strip()
            except AttributeError:
                pass
            
            gearbox = ''
            gears = ''
            engine_size = ''
            cylinders = ''
            weight = ''
            
            technical_data = soup.find('h2', class_="DetailsSectionTitle_text__KAuxN", string='Technical Data')
            if technical_data:
                technical_data_item = technical_data.find_next('div', class_="DetailsSection_childrenSection__aElbi")
                if technical_data_item:
                    gearbox_element = technical_data_item.find('dt', string='Gearbox')
                    if gearbox_element:
                        gearbox = gearbox_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
                    
                    gears_element = technical_data_item.find('dt', string='Gears')
                    if gears_element:
                        gears = gears_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
                    
                    engine_size_element = technical_data_item.find('dt', string='Engine size')
                    if engine_size_element:
                        engine_size = engine_size_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
                    
                    cylinders_element = technical_data_item.find('dt', string='Cylinders')
                    if cylinders_element:
                        cylinders = cylinders_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
                    
                    weight_element = technical_data_item.find('dt', string='Empty weight')
                    if weight_element:
                        weight = weight_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
            
            fuel = ''
            other_fuel = ''
            co2 = ''
            emission = ''
            electric_range = ''
            energy_consumption = ''
            
            consumption_data = soup.find('h2', class_="DetailsSectionTitle_text__KAuxN", string='Energy Consumption')
            if consumption_data:
                consumption_data_item = consumption_data.find_next('div', class_="DetailsSection_childrenSection__aElbi")
                if consumption_data_item:
                    
                    other_fuel_element = consumption_data_item.find('dt', string='Other fuel types')
                    if other_fuel_element:
                        other_fuel = other_fuel_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
                    
                    # fuel_item = consumption_data_item.find('dl', class_ = 'DataGrid_defaultDlStyle__xlLi_')                                             
                    # if fuel_item:
                    #      fuel_element = fuel_item.find('span', class_='DataGrid_footnote_wrapper__YGTKS')#, string='Fuel consumption')
                    #      if fuel_element:
                    #          semi_fuel = fuel_item.find_all('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01")#.text.strip()
                    #          fuel = semi_fuel[1].text.strip()
                    
                    # fuel_element = consumption_data_item.find('dt', string='Power consumption (WLTP)')
                    # if fuel_element:
                    #     fuel = fuel_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
                    
                    fuel_consumption = soup.find_all(string='Fuel consumption')
                    if fuel_consumption:
                        fuel = fuel_consumption[0].find_next('p').text.strip() 
                    
                    energy_consumption_item = soup.find_all(string='Power consumption (WLTP)') or soup.find_all(string='Power consumption')
                    if energy_consumption_item:
                       energy_consumption = energy_consumption_item[0].find_next('dd').text.strip()     
                        
                    # co2_item = consumption_data_item.find_all('dt', class_ = 'DataGrid_defaultDtStyle__soJ6R')#, string = 'CO₂-emissions (WLTP)')
                    # if len(co2_item)>=3:
                    #     co2_element = co2_item[2].find('span', class_ = 'DataGrid_footnote_wrapper__YGTKS')#, string = 'CO₂-emissions (WLTP)')
                    #     if co2_element:
                    #         co2 = co2_item[2].find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
                    
                    co2_emissions = soup.find(string='CO₂-emissions') or soup.find(string='CO₂-emissions (WLTP)') 
                    if co2_emissions:   
                       co2 = co2_emissions.find_next('dd').text.strip()
                       
                    # electric_range_element = consumption_data_item.find('dt', string='Electric Range (WLTP)')
                    # if electric_range_element:
                    #     electric_range = electric_range_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
            
                    range_item = soup.find_all(string='Electric Range (WLTP)') or soup.find_all(string='Electric Range')
                    if range_item:
                       electric_range = range_item[0].find_next('dd').text.strip() 
            
                    emission_element = consumption_data_item.find('dt', string='Emission class')
                    if emission_element:
                        emission = emission_element.find_next('dd', class_="DataGrid_defaultDdStyle__3IYpG DataGrid_fontBold__RqU01").text.strip()
                    
                        
            car_listings.append({'ID': new_id,
                                'Make': make,
                                 'Model': model,
                                 'Model Version': modelVersion,
                                 'Price': price,
                                 'Mileage': mileage,
                                 'Gearbox': gearbox,
                                 'First Registration': first_registration,
                                 'Fuel Type': fuel_type,
                                 'Power': power,
                                 'Seller': seller,
                                 'Gearbox': gearbox,
                                 'Gears': gears,
                                 'Engine Size': engine_size,
                                 'Cylinders': cylinders,
                                 'Weight': weight,
                                 'Other Fuel Types': other_fuel,
                                 'Fuel Consumption': fuel,
                                 'CO2 Emissions': co2,
                                 'Emission Class': emission,
                                 'Electric Range': electric_range,
                                 'Electric Consumption': energy_consumption,
                                 'Link': fullsite,
                                 'Data': current_date
                                })
                                 
            # Delay
            time.sleep(random.randint(1,4)) 
            print(f'Encontrados {count} carros.')
        else:
            print("Não foram encontrados mais carros!")
        pagina += 1

# Criar DF e limpeza das colunas
df = pd.DataFrame(car_listings)    
duplos = df.duplicated().sum()   
# df.drop_duplicates(subset='ID', inplace=True)
df['Site'] = 'AutoScout24'
df['Portugal'] = 'No'

print('Registos duplicados AutoScout:',duplos)

# Guardar Ficheiro CSV. 
# Obter data, nome ficheiro, caminho do ficheiro e guardar.
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
os.chdir(r'C:\Users\Rodrigo\Documents\GitHub\ImportedCars\Original Data\AutoScout24')
wd = os.getcwd()
file_name = f"Completo_AutoScout24__{current_time}.csv"
full_path = os.path.join(wd, file_name)

# Save the DataFrame to CSV with the updated file name
df.to_csv(full_path, index=False)

