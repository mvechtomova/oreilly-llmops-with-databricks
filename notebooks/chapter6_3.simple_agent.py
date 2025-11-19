# Databricks notebook source
# https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp

# https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/managed-mcp?utm_source=chatgpt.com
# genie and vector search are MCPs out of the box - no extra setup required
# add MLflow tracing!
# https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent


from databricks.sdk import WorkspaceClient
import arxiv
import json
import os
from typing import List

PAPER_DIR = "papers"
# COMMAND ----------
w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()

# COMMAND ----------
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

# COMMAND ----------
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."

# COMMAND ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for papers on arXiv based on a topic and store their information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": "Search for information about a specific paper across all topic directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The ID of the paper to look for"
                    }
                },
                "required": ["paper_id"]
            }
        }
    }
]
# COMMAND ----------
mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

def execute_tool(tool_name, tool_args):
    
    result = mapping_tool_function[tool_name](**tool_args)

    if result is None:
        result = "The operation completed but didn't return any results."
        
    elif isinstance(result, list):
        result = ', '.join(result)
        
    elif isinstance(result, dict):
        # Convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)
    
    else:
        # For any other type, convert using str()
        result = str(result)
    return result

# COMMAND ----------
def process_query(query, max_iterations=8):

    messages = [{'role': 'user', 'content': query}]

    response = openai_client.chat.completions.create(
        max_tokens=2024,
        model='databricks-gpt-oss-120b',
        tools=tools,
        messages=messages
    )

    iteration_count = 0
    process_query = True

    while process_query:
        # Safety check: prevent infinite loops
        iteration_count += 1
        if iteration_count > max_iterations:
            print(f"""\n Maximum iteration limit ({max_iterations}) reached. 
                  Sorry, I cannot help with this request 
                  as it requires too many tool calls.""")
            break

        # For OpenAI client, access the message from choices
        message = response.choices[0].message

        # Check if there are tool calls
        if message.tool_calls:
            # Add assistant message to conversation
            messages.append({
                'role': 'assistant',
                'content': message.content,
                'tool_calls': message.tool_calls
            })

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"[Iteration {iteration_count}] Calling tool {tool_name} with args {tool_args}")

                result = execute_tool(tool_name, tool_args)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result
                })

            # Get next response with tool results
            response = openai_client.chat.completions.create(
                max_tokens=2024,
                model='databricks-gpt-oss-120b',
                tools=tools,
                messages=messages
            )
        else:
            # No tool calls, just print the response and exit
            if message.content:
                print(message.content)
            process_query = False
# COMMAND ----------
def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
    
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")
# COMMAND ----------
chat_loop()
# COMMAND ----------
