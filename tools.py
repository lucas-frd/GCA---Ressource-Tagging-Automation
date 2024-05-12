import pandas as pd 
import os 
import re

df1 = pd.read_csv('/Users/lucasfernandes/Desktop/GCA - Ressource Tagging Automation/data/tools/cleaned_consolidated_tools_2023_12_10.xlsx - Sheet1.csv')
df2 = pd.read_csv('/Users/lucasfernandes/Desktop/GCA - Ressource Tagging Automation/data/tools/Resource Mapping - SMBs for DIB - Sheet1.csv')
df = pd.concat([df1, df2])
data = []

for tool in os.listdir('data/tools/exported_pages'):
    with open(f'data/tools/exported_pages/{tool}') as file:
        text = file.read()
        lines = text.split('\n')
        description = ""
        url = ""
        in_description = False
        for line in lines:
            if '<br>' in line and description == "":
                in_description = True
                continue
            if '<br>' in line and description != "":
                in_description = False
                continue
            if in_description:
                description += line.strip() + " "
            if line.startswith("https://"):
                url = line.strip()
            
        category_match = re.findall(r'\[\[Category:(.*?)\]\]', text)
        category = ", ".join(category_match).strip() if category_match else ""

        data.append({
            'Name': os.path.splitext(tool)[0],
            'Long Description': description,
            'Category': category,
            'Tool URL': url
        })

df3 = pd.DataFrame(data)
df = pd.concat([df, df3], ignore_index=True)
df = df.drop(columns=['Grouping', 'Possible Other Communities'])
df = df.sort_values(by='Name')
df = df.reset_index()
df = df.drop(df.columns[0], axis=1)

df.to_csv('Consolidated_Tools.csv')