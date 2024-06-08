import json
import pandas as pd
import uuid
df = pd.read_csv('/Users/lucasfernandes/Desktop/GCA - Ressource Tagging Automation/data/tools/Consolidated_Tools.csv')
df['Unique ID'] = [str(uuid.uuid4()) for _ in range(len(df))]
df.fillna('', inplace=True)

tools_list = []
for _, row in df.iterrows():
    tool_dict = {
        'Unique ID': row['Unique ID'], 
        'Name': row['Name'], 
        'Source': row['Source'],
        'URL': row['Tool URL'],
        'Cost': row['Cost'],
        'Certification': row['Certification'],
        'Scraped?': "No",
        'Scraped content': "",
        'Long Description': row["Long Description"],
        'Short Description': row['Short Description'], 
        'Rollover Description': row['Rollover Description'],
        'Mapped Categories': {
            'CIS v8 Control Area': {
                'High Probability': [],
                'Medium Probability': [],
                'Low Probability': [],
                'Actual': row['CIS v8 Control Area']
            }, 
            'NIST CSF Controls': {
                'High Probability': [],
                'Medium Probability': [],
                'Low Probability': [],
                'Actual': row['NIST Framework Area']
            }, 
            'Community-Specific Categories': {
                'Community': "", 
                'Categories': ""
            }, 
            'General Categories': row['Category']
            
        }
    }
    tools_list.append(tool_dict)
# Write the dictionary to a JSON file
with open('tools.json', 'w') as json_file:
    json.dump(tools_list, json_file, indent=4)
