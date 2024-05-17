from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.utils import prettify_exec_info

prompt_description_tool = "Generate a concise and informative description of the tool. Highlight its key features, functionalities, and benefits for users. Consider including information about its purpose, target audience, unique selling points, and any notable capabilities or advantages compared to similar tools in the market. The description should be clear, engaging, and compelling to potential users."
graph_config = {
   "llm": {
      "model": "ollama/llama3",
      "temperature": 1,
      "format": "json",  # Ollama needs the format to be specified explicitly
      "model_tokens": 2000, #  depending on the model set context length
      "base_url": "http://localhost:11434",  # set ollama URL of the local host (YOU CAN CHANGE IT, if you have a different endpoint
   },
   "embeddings": {
      "model": "ollama/nomic-embed-text",
      "temperature": 0,
      "base_url": "http://localhost:11434",  # set ollama URL
   }
}

# ************************************************
# Create the SmartScraperGraph instance and run it
# ************************************************
def get_tool_description(url):
   smart_scraper_graph = SmartScraperGraph(
      prompt="Generative a concise and informative description of the tool.",
      # also accepts a string with the already downloaded HTML code
      source="url",
      config=graph_config
   )
   result = smart_scraper_graph.run()
   print(result)
   return result
