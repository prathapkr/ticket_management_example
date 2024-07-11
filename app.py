import uuid
import random
from datetime import datetime, timedelta
import chromadb
from chromadb.utils import embedding_functions
import openai

# Function to generate a random date within the last 30 days
def random_date():
    return (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")

# Create dummy data with steps to resolve
dummy_tickets = [
    {
        "id": str(uuid.uuid4()),
        "title": "Server overheating",
        "description": "The main server in rack A3 is consistently running at high temperatures, causing frequent shutdowns.",
        "site": "New York Data Center",
        "category": "Hardware",
        "priority": "High",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Check and clean air vents. 2. Verify proper functioning of cooling systems. 3. Adjust server room temperature. 4. If issue persists, consider hardware upgrade or replacement."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Network connectivity issues",
        "description": "Users in the marketing department are experiencing intermittent network drops, affecting productivity.",
        "site": "London Office",
        "category": "Network",
        "priority": "Medium",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Check network switch and router configurations. 2. Verify cable connections. 3. Run network diagnostics. 4. If needed, replace faulty network equipment."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Email server down",
        "description": "The company-wide email server is not responding, affecting all departments.",
        "site": "New York Data Center",
        "category": "Software",
        "priority": "Critical",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Restart email server services. 2. Check for recent updates or changes. 3. Verify database connectivity. 4. If needed, restore from latest backup."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Printer malfunction",
        "description": "The main printer in the finance department is producing blurry prints and needs maintenance.",
        "site": "Tokyo Office",
        "category": "Hardware",
        "priority": "Low",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Clean printer heads. 2. Replace ink cartridges. 3. Run printer diagnostics. 4. If issue persists, schedule technician visit."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "VPN connection failure",
        "description": "Remote workers are unable to connect to the company VPN, hindering access to internal resources.",
        "site": "Global",
        "category": "Network",
        "priority": "High",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Verify VPN server status. 2. Check firewall rules. 3. Update VPN client software. 4. If needed, increase VPN capacity or upgrade infrastructure."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "CRM software crash",
        "description": "The customer relationship management software is crashing when generating reports.",
        "site": "London Office",
        "category": "Software",
        "priority": "Medium",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Clear application cache. 2. Verify database integrity. 3. Roll back to last stable version if recent update. 4. Contact vendor support if issue persists."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Backup failure",
        "description": "The nightly backup process failed to complete, risking data loss.",
        "site": "New York Data Center",
        "category": "Data",
        "priority": "High",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Check backup logs for errors. 2. Verify storage capacity. 3. Test backup scripts. 4. If needed, reconfigure backup jobs or upgrade storage."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Software license expiration",
        "description": "The license for our video editing software is expiring in 3 days, needed for the media team.",
        "site": "Los Angeles Office",
        "category": "Software",
        "priority": "Medium",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Contact vendor for renewal quote. 2. Process payment for license renewal. 3. Apply new license key. 4. Verify software functionality after renewal."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Security camera malfunction",
        "description": "The security camera at the main entrance is not recording properly, creating a security risk.",
        "site": "Tokyo Office",
        "category": "Security",
        "priority": "High",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Check camera power supply. 2. Verify network connectivity. 3. Test recording software. 4. If needed, replace faulty camera or schedule technician visit."
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Slow database queries",
        "description": "Users are reporting slow response times when querying the main customer database.",
        "site": "New York Data Center",
        "category": "Database",
        "priority": "Medium",
        "status": "Closed",
        "created_date": random_date(),
        "steps_to_resolve": "1. Analyze query execution plans. 2. Optimize indexes. 3. Increase database server resources if needed. 4. Consider database partitioning for large tables."
    }
]

# Set up OpenAI API key
openai.api_key = 'your-api-key-here'

# Initialize Chroma client
chroma_client = chromadb.Client()

# Create a collection for tickets
ticket_collection = chroma_client.create_collection(name="tickets")

# Set up OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key-here",
    model_name="text-embedding-ada-002"
)

def process_ticket(ticket):
    # Combine title and description for embedding
    full_text = f"{ticket['title']} {ticket['description']}"
    
    # Generate embedding
    embedding = openai_ef([full_text])[0]
    
    # Add to Chroma
    ticket_collection.add(
        ids=[ticket['id']],
        embeddings=[embedding],
        metadatas=[{
            "title": ticket['title'],
            "description": ticket['description'],
            "site": ticket['site'],
            "category": ticket['category'],
            "priority": ticket['priority'],
            "status": ticket['status'],
            "created_date": ticket['created_date'],
            "steps_to_resolve": ticket['steps_to_resolve']
        }]
    )

def find_similar_tickets_and_suggest_resolution(query_text, site, n_results=3):
    query_embedding = openai_ef([query_text])[0]
    results = ticket_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"site": site}
    )
    
    similar_tickets = []
    for i in range(len(results['ids'][0])):
        ticket_id = results['ids'][0][i]
        similarity_score = results['distances'][0][i]
        ticket_info = results['metadatas'][0][i]
        ticket_info['id'] = ticket_id
        ticket_info['similarity_score'] = similarity_score
        similar_tickets.append(ticket_info)
    
    # Generate suggested steps based on similar tickets
    suggested_steps = generate_suggested_steps(query_text, similar_tickets)
    
    return similar_tickets, suggested_steps

def generate_suggested_steps(query_text, similar_tickets):
    context = "\n".join([f"Similar ticket: {ticket['title']}\nSteps to resolve: {ticket['steps_to_resolve']}" for ticket in similar_tickets])
    
    prompt = f"""Given the following query and similar resolved tickets, suggest steps to resolve the issue:

Query: {query_text}

{context}

Suggested steps to resolve the query:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful IT support assistant that suggests resolution steps based on similar past tickets."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Process and add all dummy tickets to the database
for ticket in dummy_tickets:
    process_ticket(ticket)

# Example usage
query = "Server running slow and overheating"
site = "New York Data Center"
similar_tickets, suggested_steps = find_similar_tickets_and_suggest_resolution(query, site)

print(f"Query: {query}")
print(f"Site: {site}")
print("\nSimilar tickets:")
for ticket in similar_tickets:
    print(f"Title: {ticket['title']}")
    print(f"Description: {ticket['description']}")
    print(f"Similarity Score: {ticket['similarity_score']}")
    print(f"Steps to Resolve: {ticket['steps_to_resolve']}")
    print("---")

print("\nSuggested steps to resolve:")
print(suggested_steps)
