---
layout: post
title: Automating consumption of SDMX using Python
tags: [SDMX, Python]
excerpt_separator: <!--more-->
---
The purpose of this notebook is to describe how to create a function in Python to consume data in SDMX. Once the data is extracted it can be converted to a more user-friendly format such as Excel, csv files or it could be stored directly into a database.

<!--more-->

The first step is to import the necessary modules to create the function sdmx. The main module used in this exercise is [pandasSDMX] (https://pandasdmx.readthedocs.io/en/latest/), the rest are basic python modules.

```python
from pandasdmx import Request
import pandas as pd
import csv
import os
```
In this example, I create a function called sdmx that convert SDMX files to Excel. The same function could be used to convert files to another format or to save them into a database. The function has two arguments: country and SDMX URL.

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
Once the function is created, it can be used to convert a single SDMX file from a specific country or to loop over a list of URLs and convert several SDMX files to Excel at once. In this example, I created a file containing SDMX urls for the Consumer Price Index of Dominican Republic, Jamaica and Panama. 

```python
file=[]
f = open('SDMXlinks.csv')
csv_f = csv.reader(f)
next(csv_f)
for row in csv_f:
    file.append(row)
for country, SDMXurl in file:
    sdmx(country, SDMXurl)    
```
The Excel files can be access directly from the working directory. Here an example of each of the three files downloaded from SDMX files.

```python
dr_cpi=pd.read_excel('dominican republic.xlsx')
dr_cpi.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
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
      <th>2011-01</th>
      <th>2011-02</th>
      <th>2011-03</th>
      <th>2011-04</th>
      <th>...</th>
      <th>2018-05</th>
      <th>2018-06</th>
      <th>2018-07</th>
      <th>2018-08</th>
      <th>2018-09</th>
      <th>2018-10</th>
      <th>2018-11</th>
      <th>2018-12</th>
      <th>2019-01</th>
      <th>2019-02</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CPI</td>
      <td>DO</td>
      <td>PCPI_IX</td>
      <td>Consumer Price Index, All items</td>
      <td>_Z</td>
      <td>M</td>
      <td>101.2437</td>
      <td>102.461</td>
      <td>103.645</td>
      <td>104.55</td>
      <td>...</td>
      <td>129.7</td>
      <td>129.97</td>
      <td>129.95</td>
      <td>129.99</td>
      <td>130.09</td>
      <td>130.38</td>
      <td>129.92</td>
      <td>129.64</td>
      <td>129.42</td>
      <td>129.9</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 104 columns</p>
</div>




```python
jamaica_cpi=pd.read_excel('jamaica.xlsx')
jamaica_cpi.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
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
      <th>2016-06</th>
      <th>2016-07</th>
      <th>2016-08</th>
      <th>2016-09</th>
      <th>...</th>
      <th>2018-05</th>
      <th>2018-06</th>
      <th>2018-07</th>
      <th>2018-08</th>
      <th>2018-09</th>
      <th>2018-10</th>
      <th>2018-11</th>
      <th>2018-12</th>
      <th>2019-01</th>
      <th>2019-02</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CPI</td>
      <td>JM</td>
      <td>PCPI_IX</td>
      <td>Consumer Price Index, All items</td>
      <td>_Z</td>
      <td>M</td>
      <td>231.0</td>
      <td>232.1</td>
      <td>233.1</td>
      <td>234.2</td>
      <td>...</td>
      <td>246.97664</td>
      <td>248.010975</td>
      <td>250.446135</td>
      <td>252.776626</td>
      <td>255.578838</td>
      <td>257.426564</td>
      <td>257.387725</td>
      <td>254.741694</td>
      <td>254.182631</td>
      <td>254.3392</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 39 columns</p>
</div>




```python
panama_cpi=pd.read_excel('panama.xlsx')
panama_cpi.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
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
      <td>105.527</td>
      <td>105.4892</td>
      <td>105.5743</td>
      <td>105.1135</td>
      <td>104.6655</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 201 columns</p>
</div>

