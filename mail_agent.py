import os
from langchain_community.agent_toolkits.gmail.toolkit import GmailToolkit
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
import json
import re

# ✅ Step 1: Load Gmail Toolkit with credentials from environment
toolkit = GmailToolkit(credentials_path=os.getenv("GMAIL_CREDENTIALS_PATH"))
tools = toolkit.get_tools()

# Get direct access to the create_draft tool for fallback use
create_draft_tool = None
for tool in tools:
    if tool.name == "create_gmail_draft":
        create_draft_tool = tool
        break

# ✅ Step 2: Initialize Groq LLaMA3 LLM using env var for API key
llm = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ✅ Step 3: Initialize Agent with parsing error handling
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=10,
    max_execution_time=300,
    handle_parsing_errors=True,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)

# ✅ Step 4: Run agent with safe prompt formatting and manual execution if needed
def runAgent(task):
    try:
        print("\n[⏳ Agent Running...]\n")
        formatted_task = f"""
Please help with the following email task:
{task}

If you need to draft an email, provide the details in this exact format:
Action: create_gmail_draft
Action Input: {{"to": "recipient@example.com", "subject": "Meeting Tomorrow", "message": "Hello, looking forward to our meeting."}}

Return a Final Answer when complete.
"""
        response = agent.invoke({"input": formatted_task})
        
        if isinstance(response, dict) and "output" in response:
            output = response["output"]
            
            action_match = re.search(r"Action: create_gmail_draft\s+Action Input: ({.*})", output, re.DOTALL)
            if action_match:
                try:
                    action_input_str = action_match.group(1).strip()
                    action_input = json.loads(action_input_str)
                    
                    print("\n[⚠️ Agent created plan but didn't execute - manually executing create_gmail_draft...]")
                    
                    if create_draft_tool:
                        draft_result = create_draft_tool.invoke(action_input)
                        print(f"\n[✅ Draft creation result]: {draft_result}")
                        response["output"] += "\n\n[Draft has been successfully created in your Gmail account]"
                except json.JSONDecodeError:
                    print(f"\n[❌ Failed to parse action input: {action_input_str}]")
                except Exception as e:
                    print(f"\n[❌ Failed to execute create_gmail_draft: {e}]")
        
        return response
    except Exception as e:
        print(f"\n[❌ Exception Occurred]: {e}")
        return None

# Helper function to manually create drafts
def create_draft_manually(to, subject, message):
    if create_draft_tool:
        try:
            result = create_draft_tool.invoke({
                "to": to,
                "subject": subject,
                "message": message
            })
            print(f"\n[✅ Manual draft creation result]: {result}")
            return True
        except Exception as e:
            print(f"\n[❌ Manual draft creation failed]: {e}")
            return False
    else:
        print("\n[❌ Create draft tool not found]")
        return False

# ✅ Step 5: Execution entry point
if __name__ == "__main__":
    print(f"Loaded tools: {[tool.name for tool in tools]}")
    print("\nℹ️ Example command: Draft an email to john@example.com with subject 'Happy Birthday' and message 'Have a great day!'\n")

    while True:
        task = input("Hello Sir, how can I help you with emails today? (or type 'exit' to quit): \n")
        
        if task.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
            
        if "draft" in task.lower() and "manually" in task.lower():
            print("\n[📝 Manual Draft Mode]")
            to = input("To: ")
            subject = input("Subject: ")
            message = input("Message (type 'END' on a new line when finished):\n")
            
            message_lines = []
            line = input()
            while line != "END":
                message_lines.append(line)
                line = input()
            
            full_message = "\n".join(message_lines)
            result = create_draft_manually(to, subject, full_message)
            if result:
                print("\n[✅ Draft created successfully]")
            continue

        response = runAgent(task)

        if not response:
            print("[⚠️ No response from agent — trying Gmail tools directly as fallback...]")
            
            if any(word in task.lower() for word in ["draft", "email", "write", "compose"]):
                print("\n[📝 Let's create a draft manually as fallback]")
                to = input("To: ")
                subject = input("Subject: ")
                message = input("Message (type 'END' on a new line when finished):\n")
                
                message_lines = []
                line = input()
                while line != "END":
                    message_lines.append(line)
                    line = input()
                
                full_message = "\n".join(message_lines)
                result = create_draft_manually(to, subject, full_message)
            else:
                try:
                    for tool in tools:
                        if tool.name == "search_gmail":
                            query = 'newer_than:7d'
                            results = tool.invoke({"query": query})
                            print("[📨 Fallback Gmail Tool Output]:\n", results)
                            break
                except Exception as fallback_error:
                    print("\n[🔥 Fallback Tool Error]:")
                    print(fallback_error)
                    print("\n👉 Make sure Gmail API is enabled for your project.")
                    print("🔗 Visit: https://console.developers.google.com/apis/api/gmail.googleapis.com/overview")
        else:
            print("\n[✅ Agent Response]:\n")
            print(response["output"] if isinstance(response, dict) and "output" in response else response)
            
        print("\n" + "-"*50 + "\n")
print("[⚠️ An unexpected error occurred]")