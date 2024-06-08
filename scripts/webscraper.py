from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.utils import prettify_exec_info

prompt_description_tool = "Generate a concise and informative description of the tool. Highlight its key features, functionalities, and benefits for users. Consider including information about its purpose, target audience, unique selling points, and any notable capabilities or advantages compared to similar tools in the market. The description should be clear, engaging, and compelling to potential users."
graph_config = {
    "llm": {
        "api_key": 'sk-resource-tagging-automation-q9HPbyzGtZCZnmJuuyMET3BlbkFJycWpNh4CbwKCDoktRJAD',
        "model": "gpt-3.5-turbo",
        "temperature":0,
    },
    "verbose":True,
}

# ************************************************
# Create the SmartScraperGraph instance and run it
# ************************************************
def get_tool_description(url):
   smart_scraper_graph = SmartScraperGraph(
      prompt='Generate a concise and informative description of the tool',
      # also accepts a string with the already downloaded HTML code
      source=url,
      config=graph_config
   )
   result = smart_scraper_graph.run()
   print(result)
   input_tokens = 0.25 * len(prompt_description_tool)
   output_tokens = 0.25 * len(result)
   input_cost = input_tokens * 0.5 / 1000000
   output_cost = output_tokens * 1.5 / 1000000
   total_cost = input_cost + output_cost
   print(f'total cost = {total_cost}')
   return result



get_tool_description('https://1fort.com/')

input_tokens = 0.25 * len(prompt_description_tool)
output_tokens = 0.25 * len(result)