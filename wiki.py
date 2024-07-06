import wikipediaapi
import json

def fetch_wikipedia_data(query, lang='en'):
    wiki_wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent='YourAppName (https://yourappwebsite.com/) Contact at your-email@example.com'
    )
    
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary
    else:
        return None

queries = ["President of the United States", "Vice President of the United States", "United States", "Joe Biden", "Kamala Harris"]
data = []

for query in queries:
    summary = fetch_wikipedia_data(query)
    if summary:
        data.append({
            "context": summary,
            "question": f"What is {query}?",
            "answer": summary.split('.')[0]  # Taking the first sentence as the answer
        })

# Save data to a JSON file
with open('wikipedia_data.json', 'w') as f:
    json.dump(data, f)
