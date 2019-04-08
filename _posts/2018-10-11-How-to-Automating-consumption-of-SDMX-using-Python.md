---
layout: post
title: Automating consumption of SDMX using Python
tags: [SDMX, Python]
excerpt_separator: <!--more-->
---
This notebook is to describe how to create a function in Python to consume SDMX files. The data can be then converted to Excel files or save in a database.

<!--more-->

```python
from pandasdmx import Request
import pandas as pd
import csv
import os
import warnings
warnings.filterwarnings('ignore')

```


```python
def sdmx(country, SDMXurl):
     
    '''Create function to convert SDMX files to Excel'''
   
    IMF = Request ("IMF_SDMXCENTRAL")
    data_resp = IMF.data(url=SDMXurl)
    data_frame = data_resp.write(data_resp.data.series, dtype = str, asframe = True, parse_time=False)
    attributes_frame = data_resp.write(data_resp.data.series, dtype = str, asframe = True, attributes = "s", parse_time=False)
    
    #One time only. Once the the ECOFIN_CL is pulled from SDMX Central it could be used for all data categories.
    IMF_flow = IMF.dataflow('CPI')
    dsd = IMF_flow.dataflow.CPI.structure()
    ECOFIN_CL = IMF_flow.write().codelist.loc['CL_INDICATOR']


    data = data_frame.transpose()
    data = data.reset_index()
    data = pd.merge(data, ECOFIN_CL, how = 'left', left_on = "INDICATOR",  right_index=True)

    data.insert(3, "DESCRIPTOR", data.name)
    data.drop(labels = "name", axis = 1, inplace = True)

    #Save to Excel but it could also be pushed to a database
    wd = os.getcwd()
    data.to_excel('%s.xlsx' % country)
```


```python
file=[]
f = open('SDMXlinks.csv')
csv_f = csv.reader(f)
next(csv_f)
for row in csv_f:
    file.append(row)
    
```


```python
for country, SDMXurl in file:
    sdmx(country, SDMXurl)
```


```python
panama_cpi=pd.read_excel('panama.xlsx')
panama_cpi.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATA_DOMAIN</th>
      <th>REF_AREA</th>
      <th>INDICATOR</th>
      <th>DESCRIPTOR</th>
      <th>COUNTERPART_AREA</th>
      <th>FREQ</th>
      <th>2002-10</th>
      <th>2002-11</th>
      <th>2002-12</th>
      <th>2003-01</th>
      <th>...</th>
      <th>2018-03</th>
      <th>2018-04</th>
      <th>2018-05</th>
      <th>2018-06</th>
      <th>2018-07</th>
      <th>2018-08</th>
      <th>2018-09</th>
      <th>2018-10</th>
      <th>2018-11</th>
      <th>2018-12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CPI</td>
      <td>PA</td>
      <td>PCPI_IX</td>
      <td>Consumer Price Index, All items</td>
      <td>_Z</td>
      <td>M</td>
      <td>67.523849</td>
      <td>67.591372</td>
      <td>67.456325</td>
      <td>67.591372</td>
      <td>...</td>
      <td>105.1</td>
      <td>105.3</td>
      <td>105.3</td>
      <td>105.5165</td>
      <td>105.4581</td>
      <td>105.5270</td>
      <td>105.4892</td>
      <td>105.5743</td>
      <td>105.1135</td>
      <td>104.6655</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CPI</td>
      <td>PA</td>
      <td>PCPI_CP_01_IX</td>
      <td>Prices, Consumer Price Index, Food and non-alc...</td>
      <td>_Z</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>101.2</td>
      <td>101.2</td>
      <td>101.0</td>
      <td>100.9833</td>
      <td>101.2679</td>
      <td>101.5793</td>
      <td>101.5739</td>
      <td>101.8383</td>
      <td>101.8911</td>
      <td>102.1452</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CPI</td>
      <td>PA</td>
      <td>PCPI_CP_02_IX</td>
      <td>Prices, Consumer Price Index, Alcoholic Bevera...</td>
      <td>_Z</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>109.6</td>
      <td>110.1</td>
      <td>110.3</td>
      <td>110.3543</td>
      <td>110.6632</td>
      <td>110.8176</td>
      <td>111.4663</td>
      <td>111.6671</td>
      <td>110.9875</td>
      <td>108.9024</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CPI</td>
      <td>PA</td>
      <td>PCPI_CP_03_IX</td>
      <td>Prices, Consumer Price Index, Clothing and foo...</td>
      <td>_Z</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>96.9</td>
      <td>96.9</td>
      <td>96.4</td>
      <td>96.3397</td>
      <td>96.3293</td>
      <td>96.3827</td>
      <td>95.6729</td>
      <td>95.6625</td>
      <td>95.5935</td>
      <td>95.6365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CPI</td>
      <td>PA</td>
      <td>PCPI_CP_04_IX</td>
      <td>Prices, Consumer Price Index, Housing, Water, ...</td>
      <td>_Z</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>101.6</td>
      <td>101.6</td>
      <td>101.8</td>
      <td>101.8034</td>
      <td>102.2432</td>
      <td>102.3326</td>
      <td>102.3197</td>
      <td>102.3185</td>
      <td>102.0680</td>
      <td>101.8892</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 201 columns</p>
</div>
