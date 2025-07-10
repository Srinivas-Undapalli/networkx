import oracledb
import asyncio
import pandas as pd
from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, END
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State passed between nodes"""
    query_result: Any
    pandas_result: Any
    user_id: str
    error: str
    db_connection: Any  # Database connection for the entire agent execution

class DatabaseService:
    """Database connection pool manager"""
    
    def __init__(self):
        self.pool = None
    
    async def initialize_pool(self, user: str, password: str, dsn: str, 
                            min_connections: int = 2, max_connections: int = 10):
        """Initialize the connection pool"""
        try:
            self.pool = await oracledb.create_pool_async(
                user=user,
                password=password,
                dsn=dsn,
                min=min_connections,
                max=max_connections,
                increment=1,
                threaded=True,
                getmode=oracledb.POOL_GETMODE_WAIT,
                timeout=30  # seconds to wait for connection
            )
            logger.info(f"Connection pool initialized with {min_connections}-{max_connections} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def get_connection(self):
        """Get connection from pool"""
        if not self.pool:
            raise Exception("Connection pool not initialized")
        return await self.pool.acquire()
    
    async def close_pool(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Connection pool closed")

# Global database service instance
db_service = DatabaseService()

async def acquire_connection_node(state: AgentState) -> AgentState:
    """First node: Acquire database connection for the entire agent execution"""
    try:
        connection = await db_service.get_connection()
        state['db_connection'] = connection
        logger.info("Database connection acquired for agent execution")
    except Exception as e:
        error_msg = f"Failed to acquire database connection: {str(e)}"
        logger.error(error_msg)
        state['error'] = error_msg
        state['db_connection'] = None
    
    return state

async def pandas_query_node(state: AgentState) -> AgentState:
    """Execute pandas query using existing connection"""
    if state.get('error') or not state.get('db_connection'):
        logger.warning("Skipping pandas query due to connection error")
        return state
    
    try:
        connection = state['db_connection']
        
        # Execute query using pandas
        query = "SELECT user_id, username, email FROM users WHERE user_id = :user_id"
        df = pd.read_sql(query, connection, params={'user_id': state['user_id']})
        
        # Store results
        state['query_result'] = df.to_dict('records')
        state['pandas_result'] = df
        logger.info(f"Pandas query executed successfully, found {len(df)} records")
        
    except Exception as e:
        error_msg = f"Pandas query failed: {str(e)}"
        logger.error(error_msg)
        state['error'] = error_msg
    
    return state

async def raw_query_node(state: AgentState) -> AgentState:
    """Execute raw query using existing connection"""
    if state.get('error') or not state.get('db_connection'):
        logger.warning("Skipping raw query due to connection error")
        return state
    
    try:
        connection = state['db_connection']
        
        # Execute raw query
        async with connection.cursor() as cursor:
            # Example: Update last login
            update_query = "UPDATE users SET last_login = SYSDATE WHERE user_id = :user_id"
            await cursor.execute(update_query, {'user_id': state['user_id']})
            
            # Commit the transaction
            await connection.commit()
            logger.info("Raw query executed and committed successfully")
            
    except Exception as e:
        error_msg = f"Raw query failed: {str(e)}"
        logger.error(error_msg)
        state['error'] = error_msg
        # Rollback on error
        try:
            await state['db_connection'].rollback()
            logger.info("Transaction rolled back due to error")
        except:
            pass
    
    return state

async def multiple_operations_node(state: AgentState) -> AgentState:
    """Execute multiple database operations using the same connection"""
    if state.get('error') or not state.get('db_connection'):
        logger.warning("Skipping multiple operations due to connection error")
        return state
    
    try:
        connection = state['db_connection']
        
        # Operation 1: Raw query for insert
        async with connection.cursor() as cursor:
            insert_query = """
                INSERT INTO user_activity (user_id, activity_type, activity_date) 
                VALUES (:user_id, 'agent_execution', SYSDATE)
            """
            await cursor.execute(insert_query, {'user_id': state['user_id']})
        
        # Operation 2: Pandas query for analysis
        analytics_query = """
            SELECT activity_type, COUNT(*) as count 
            FROM user_activity 
            WHERE user_id = :user_id 
            GROUP BY activity_type
        """
        analytics_df = pd.read_sql(analytics_query, connection, params={'user_id': state['user_id']})
        
        # Operation 3: Another raw query
        async with connection.cursor() as cursor:
            update_query = "UPDATE users SET profile_updated = SYSDATE WHERE user_id = :user_id"
            await cursor.execute(update_query, {'user_id': state['user_id']})
        
        # Commit all operations together
        await connection.commit()
        
        # Store analytics results
        state['analytics_result'] = analytics_df.to_dict('records')
        logger.info("Multiple operations executed successfully in single transaction")
        
    except Exception as e:
        error_msg = f"Multiple operations failed: {str(e)}"
        logger.error(error_msg)
        state['error'] = error_msg
        # Rollback all operations
        try:
            await state['db_connection'].rollback()
            logger.info("All operations rolled back due to error")
        except:
            pass
    
    return state

async def process_results_node(state: AgentState) -> AgentState:
    """Process the results from database operations"""
    if state.get('error'):
        logger.error(f"Processing skipped due to error: {state['error']}")
        return state
    
    if state.get('query_result'):
        logger.info(f"Processing {len(state['query_result'])} records")
        # Add your result processing logic here
    
    return state

async def release_connection_node(state: AgentState) -> AgentState:
    """Final node: Release the database connection back to pool"""
    try:
        if state.get('db_connection'):
            await state['db_connection'].close()
            state['db_connection'] = None
            logger.info("Database connection released back to pool")
    except Exception as e:
        logger.error(f"Error releasing connection: {e}")
    
    return state

def create_agent_graph():
    """Create the LangGraph workflow with connection management"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("acquire_connection", acquire_connection_node)
    workflow.add_node("pandas_query", pandas_query_node)
    workflow.add_node("raw_query", raw_query_node)
    workflow.add_node("multiple_operations", multiple_operations_node)
    workflow.add_node("process_results", process_results_node)
    workflow.add_node("release_connection", release_connection_node)
    
    # Define edges - connection management at start and end
    workflow.set_entry_point("acquire_connection")
    workflow.add_edge("acquire_connection", "pandas_query")
    workflow.add_edge("pandas_query", "raw_query")
    workflow.add_edge("raw_query", "multiple_operations")
    workflow.add_edge("multiple_operations", "process_results")
    workflow.add_edge("process_results", "release_connection")
    workflow.add_edge("release_connection", END)
    
    return workflow.compile()

# Context manager approach for the entire agent execution
class AgentDatabaseContext:
    """Context manager for entire agent execution"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.connection = None
    
    async def __aenter__(self):
        self.connection = await self.db_service.get_connection()
        logger.info("Database connection acquired for agent context")
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            try:
                if exc_type:
                    await self.connection.rollback()
                    logger.info("Transaction rolled back due to exception")
                await self.connection.close()
                logger.info("Database connection released from agent context")
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")

# Alternative: Simplified agent execution with context manager
async def execute_agent_with_context(user_id: str):
    """Execute agent operations using context manager"""
    try:
        async with AgentDatabaseContext(db_service) as connection:
            # All database operations using the same connection
            
            # Pandas query
            user_query = "SELECT user_id, username, email FROM users WHERE user_id = :user_id"
            user_df = pd.read_sql(user_query, connection, params={'user_id': user_id})
            
            # Raw query
            async with connection.cursor() as cursor:
                update_query = "UPDATE users SET last_login = SYSDATE WHERE user_id = :user_id"
                await cursor.execute(update_query, {'user_id': user_id})
            
            # Another pandas query
            activity_query = "SELECT * FROM user_activity WHERE user_id = :user_id ORDER BY activity_date DESC"
            activity_df = pd.read_sql(activity_query, connection, params={'user_id': user_id})
            
            # Commit all operations
            await connection.commit()
            
            return {
                'user_data': user_df.to_dict('records'),
                'activity_data': activity_df.to_dict('records')
            }
            
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise

async def main():
    """Example usage with connection per agent execution"""
    try:
        # Initialize database connection pool
        await db_service.initialize_pool(
            user="your_username",
            password="your_password",
            dsn="your_host:1521/your_service_name",
            min_connections=5,
            max_connections=20
        )
        
        # Method 1: Using LangGraph with connection management nodes
        agent = create_agent_graph()
        
        # Execute multiple agents concurrently
        tasks = []
        for i in range(10):  # 10 concurrent agent executions
            initial_state = {
                'user_id': f'user_{i}',
                'query_result': None,
                'pandas_result': None,
                'error': None,
                'db_connection': None
            }
            task = agent.ainvoke(initial_state)
            tasks.append(task)
        
        # Wait for all agent executions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Print results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} failed: {result}")
            else:
                logger.info(f"Agent {i} completed successfully")
        
        # Method 2: Using context manager approach
        context_tasks = []
        for i in range(5):
            task = execute_agent_with_context(f'user_context_{i}')
            context_tasks.append(task)
        
        context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
        
        for i, result in enumerate(context_results):
            if isinstance(result, Exception):
                logger.error(f"Context agent {i} failed: {result}")
            else:
                logger.info(f"Context agent {i} completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
    
    finally:
        # Close connection pool
        await db_service.close_pool()

if __name__ == "__main__":
    asyncio.run(main())
