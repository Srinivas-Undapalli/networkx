from typing import Dict, List, Any, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
import sqlite3
import json
from datetime import datetime

# Database tool
@tool
def query_database(query: str) -> str:
    """Execute a SQL query and return results as JSON string"""
    try:
        conn = sqlite3.connect('customer_data.db')  # Replace with your DB
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        result = {
            "success": True,
            "data": df.to_dict('records'),
            "columns": list(df.columns),
            "row_count": len(df)
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": [],
            "columns": [],
            "row_count": 0
        })

@tool
def analyze_data(data_json: str, table_context: str, column_context: str) -> str:
    """Analyze query results and extract meaningful context"""
    try:
        data = json.loads(data_json)
        if not data["success"]:
            return json.dumps({"error": "No data to analyze"})
        
        df = pd.DataFrame(data["data"])
        column_info = json.loads(column_context) if column_context else {}
        
        # Basic analysis
        analysis = {
            "table_purpose": table_context,
            "record_count": len(df),
            "insights": [],
            "column_analysis": {},
            "summary_metrics": {}
        }
        
        # Generate insights
        if len(df) == 0:
            analysis["insights"].append("No records found for this customer")
        else:
            analysis["insights"].append(f"Found {len(df)} records")
            
            # Analyze key columns
            for col in df.columns:
                if col in column_info:
                    purpose = column_info[col]
                    col_analysis = {"purpose": purpose}
                    
                    if "status" in purpose.lower():
                        col_analysis["distribution"] = df[col].value_counts().to_dict()
                        if "failed" in str(df[col].values):
                            failed_count = len(df[df[col] == "failed"])
                            analysis["insights"].append(f"Found {failed_count} failed records")
                    
                    elif "amount" in purpose.lower():
                        col_analysis["total"] = float(df[col].sum())
                        col_analysis["average"] = float(df[col].mean())
                        analysis["insights"].append(f"Total {col}: {col_analysis['total']}")
                    
                    elif "date" in purpose.lower():
                        try:
                            dates = pd.to_datetime(df[col])
                            latest = dates.max()
                            days_ago = (datetime.now() - latest).days
                            analysis["insights"].append(f"Last activity: {days_ago} days ago")
                        except:
                            pass
                    
                    analysis["column_analysis"][col] = col_analysis
        
        return json.dumps(analysis)
    except Exception as e:
        return json.dumps({"error": str(e)})

class TableAnalysisSupervisor:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [query_database, analyze_data]
        
        # Create React agent
        self.agent = create_react_agent(
            model=llm,
            tools=self.tools,
            checkpointer=MemorySaver()
        )
        
        # System prompt for the agent
        self.system_prompt = """
        You are a database analysis agent specialized in customer fallout analysis.
        
        Your task:
        1. Execute SQL queries to get customer data from a specific table
        2. Analyze the results to extract meaningful context
        3. Provide insights about potential customer fallout indicators
        
        Always follow this process:
        1. Use query_database tool to get data
        2. Use analyze_data tool to extract context
        3. Summarize findings in a clear format
        
        Be concise and focus on actionable insights.
        """
    
    async def analyze_table(self, customer_id: str, table_name: str, 
                           table_context: str, column_context: Dict[str, str] = None) -> Dict[str, Any]:
        """Analyze a table for customer fallout context"""
        
        # Prepare the analysis request
        column_context_json = json.dumps(column_context or {})
        
        user_message = f"""
        Analyze table '{table_name}' for customer '{customer_id}'.
        
        Table Context: {table_context}
        Column Context: {column_context_json}
        
        Steps:
        1. Query the table for this customer's data
        2. Analyze the results using the provided context
        3. Extract key insights about potential customer issues
        
        Start by querying: SELECT * FROM {table_name} WHERE customer_id = '{customer_id}' ORDER BY created_date DESC LIMIT 50
        """
        
        # Create thread config
        config = {"configurable": {"thread_id": f"analysis_{customer_id}_{table_name}"}}
        
        # Run the agent
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]
        
        try:
            result = await self.agent.ainvoke(
                {"messages": messages}, 
                config=config
            )
            
            # Extract the final analysis from agent response
            final_message = result["messages"][-1].content
            
            # Try to parse JSON from the response
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', final_message, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    # Fallback to structured response
                    analysis_data = {
                        "table_purpose": table_context,
                        "insights": [final_message],
                        "status": "completed"
                    }
            except:
                analysis_data = {
                    "table_purpose": table_context,
                    "insights": [final_message],
                    "status": "completed"
                }
            
            return {
                "customer_id": customer_id,
                "table_name": table_name,
                "status": "success",
                "context": analysis_data,
                "agent_response": final_message
            }
            
        except Exception as e:
            return {
                "customer_id": customer_id,
                "table_name": table_name,
                "status": "error",
                "error": str(e),
                "context": {}
            }

# Alternative: Simple Supervisor using StateGraph
class SimpleTableSupervisor:
    def __init__(self, llm):
        self.llm = llm
        self.graph = self._create_graph()
    
    def _create_graph(self):
        def query_step(state: MessagesState):
            # Extract query info from the last message
            last_message = state["messages"][-1].content
            
            # Simple query extraction (you can make this smarter)
            if "customer_id:" in last_message and "table:" in last_message:
                lines = last_message.split('\n')
                customer_id = next((line.split('customer_id:')[1].strip() for line in lines if 'customer_id:' in line), '')
                table_name = next((line.split('table:')[1].strip() for line in lines if 'table:' in line), '')
                
                query = f"SELECT * FROM {table_name} WHERE customer_id = '{customer_id}' LIMIT 50"
                result = query_database(query)
                
                return {"messages": [HumanMessage(content=f"Query result: {result}")]}
            
            return {"messages": [HumanMessage(content="Could not extract query parameters")]}
        
        def analyze_step(state: MessagesState):
            # Get the query result from previous step
            last_message = state["messages"][-1].content
            
            if "Query result:" in last_message:
                result_json = last_message.replace("Query result: ", "")
                analysis = analyze_data(result_json, "Customer data analysis", "{}")
                
                return {"messages": [HumanMessage(content=f"Analysis: {analysis}")]}
            
            return {"messages": [HumanMessage(content="No query result to analyze")]}
        
        # Create graph
        workflow = StateGraph(MessagesState)
        workflow.add_node("query", query_step)
        workflow.add_node("analyze", analyze_step)
        
        workflow.set_entry_point("query")
        workflow.add_edge("query", "analyze")
        workflow.add_edge("analyze", "__end__")
        
        return workflow.compile()
    
    async def analyze(self, customer_id: str, table_name: str, table_context: str):
        """Simple analysis using StateGraph"""
        initial_message = f"""
        customer_id: {customer_id}
        table: {table_name}
        context: {table_context}
        """
        
        result = await self.graph.ainvoke({
            "messages": [HumanMessage(content=initial_message)]
        })
        
        return result["messages"][-1].content

# Usage Examples
async def main():
    # Option 1: Using React Agent (Recommended)
    print("🤖 Using React Agent Supervisor")
    
    # Replace with your actual LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    supervisor = TableAnalysisSupervisor(llm)
    
    # Define context
    table_context = "Customer transaction records showing payment history and failures"
    column_context = {
        "customer_id": "Unique customer identifier",
        "amount": "Transaction amount in USD",
        "status": "Transaction status (success/failed/pending)",
        "transaction_date": "Date and time of transaction",
        "failure_reason": "Reason for transaction failure"
    }
    
    # Analyze
    result = await supervisor.analyze_table(
        customer_id="CUST_12345",
        table_name="transactions",
        table_context=table_context,
        column_context=column_context
    )
    
    print("📊 Results:")
    print(json.dumps(result, indent=2))
    
    # Option 2: Using Simple StateGraph
    print("\n🔄 Using Simple StateGraph Supervisor")
    
    simple_supervisor = SimpleTableSupervisor(llm)
    simple_result = await simple_supervisor.analyze(
        customer_id="CUST_12345",
        table_name="transactions", 
        table_context=table_context
    )
    
    print("📈 Simple Results:")
    print(simple_result)

# Minimal usage function
async def quick_analyze(customer_id: str, table_name: str, table_context: str, llm):
    """Quick one-liner analysis"""
    supervisor = TableAnalysisSupervisor(llm)
    return await supervisor.analyze_table(customer_id, table_name, table_context)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
