import pandas as pd
import multiprocessing as mp
from functools import partial
import time
from typing import List, Dict, Any, Union
import tempfile
import os

class SimplifiedExcelProcessor:
    """
    Simplified Excel processor that handles multiple output rows per input row.
    Perfect for scenarios where one input generates multiple results.
    """
    
    def __init__(self, input_file: str, output_file: str, chunk_size: int = 500):
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.num_processes = mp.cpu_count() - 1
    
    def process_single_row(self, row_data: Dict[str, Any], script_func) -> List[Dict[str, Any]]:
        """
        Process a single input row and return multiple output rows.
        
        Args:
            row_data: Input row as dictionary
            script_func: Your processing function
            
        Returns:
            List of output rows (can be 1 or many rows)
        """
        try:
            # Call your processing function
            results = script_func(row_data)
            
            # Handle different return types from your function
            if results is None:
                return [self._create_error_row(row_data, "Function returned None")]
            
            # If single result, convert to list
            if not isinstance(results, list):
                results = [results]
            
            output_rows = []
            for i, result in enumerate(results):
                # Create output row with original input data + result
                output_row = row_data.copy()  # Keep original columns
                
                # Add result data
                if isinstance(result, dict):
                    # If result is dict, add all key-value pairs
                    output_row.update(result)
                else:
                    # If result is single value, add as 'result' column
                    output_row['result'] = result
                
                # Add metadata
                output_row['input_row_id'] = row_data.get('row_id', id(row_data))
                output_row['output_sequence'] = i + 1
                output_row['status'] = 'success'
                output_row['error'] = None
                
                output_rows.append(output_row)
            
            return output_rows
            
        except Exception as e:
            # Return error row
            return [self._create_error_row(row_data, str(e))]
    
    def _create_error_row(self, row_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Create an error row when processing fails."""
        error_row = row_data.copy()
        error_row['input_row_id'] = row_data.get('row_id', id(row_data))
        error_row['output_sequence'] = 1
        error_row['status'] = 'failed'
        error_row['error'] = error_msg
        error_row['result'] = None
        return error_row
    
    def process_chunk(self, chunk_df: pd.DataFrame, script_func) -> pd.DataFrame:
        """Process a chunk of rows in parallel."""
        # Add row_id for tracking
        chunk_df = chunk_df.reset_index()
        chunk_df['row_id'] = chunk_df.index
        
        # Convert to list of dictionaries
        input_rows = chunk_df.to_dict('records')
        
        # Process in parallel
        process_func = partial(self.process_single_row, script_func=script_func)
        
        with mp.Pool(processes=self.num_processes) as pool:
            results_list = pool.map(process_func, input_rows)
        
        # Flatten results (each input row can produce multiple output rows)
        all_output_rows = []
        for row_results in results_list:
            all_output_rows.extend(row_results)
        
        return pd.DataFrame(all_output_rows)
    
    def run(self, script_func) -> pd.DataFrame:
        """
        Main processing method.
        
        Args:
            script_func: Your processing function that takes a row dict and 
                        returns either a single result, dict, or list of results
        
        Returns:
            Final DataFrame with all results
        """
        print(f"Starting processing of {self.input_file}...")
        start_time = time.time()
        
        temp_files = []
        chunk_count = 0
        total_input_rows = 0
        total_output_rows = 0
        
        try:
            # Process in chunks
            for chunk_df in pd.read_excel(self.input_file, chunksize=self.chunk_size):
                chunk_count += 1
                input_rows_count = len(chunk_df)
                total_input_rows += input_rows_count
                
                print(f"Processing chunk {chunk_count} ({input_rows_count} input rows)...")
                
                # Process chunk
                result_chunk = self.process_chunk(chunk_df, script_func)
                output_rows_count = len(result_chunk)
                total_output_rows += output_rows_count
                
                print(f"  → Generated {output_rows_count} output rows")
                
                # Save chunk to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                result_chunk.to_excel(temp_file.name, index=False)
                temp_files.append(temp_file.name)
                temp_file.close()
            
            # Combine all chunks
            print("Combining results...")
            combined_results = []
            for temp_file in temp_files:
                chunk_result = pd.read_excel(temp_file)
                combined_results.append(chunk_result)
            
            final_df = pd.concat(combined_results, ignore_index=True)
            
            # Save final result
            final_df.to_excel(self.output_file, index=False)
            
            # Print summary
            processing_time = time.time() - start_time
            successful_rows = (final_df['status'] == 'success').sum()
            failed_rows = (final_df['status'] == 'failed').sum()
            
            print(f"\nProcessing Complete!")
            print(f"Input rows: {total_input_rows}")
            print(f"Output rows: {total_output_rows}")
            print(f"Successful: {successful_rows}")
            print(f"Failed: {failed_rows}")
            print(f"Expansion ratio: {total_output_rows/total_input_rows:.2f}x")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Results saved to: {self.output_file}")
            
            return final_df
            
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

# =============================================================================
# EXAMPLE USAGE SCENARIOS
# =============================================================================

def example_single_result_function(row_data):
    """
    Example 1: Single result per input row (1:1 mapping)
    """
    input_value = row_data.get('input_column', 0)
    result = input_value * 2
    return result

def example_multiple_results_function(row_data):
    """
    Example 2: Multiple results per input row (1:many mapping)
    Your function returns a list of results
    """
    input_value = row_data.get('input_column', 0)
    
    # Generate multiple results for this input
    results = []
    for i in range(3):  # Generate 3 results per input
        result = input_value * (i + 1)
        results.append(result)
    
    return results

def example_dict_results_function(row_data):
    """
    Example 3: Multiple columns as output
    Your function returns a dictionary or list of dictionaries
    """
    param1 = row_data.get('param1', 0)
    param2 = row_data.get('param2', 0)
    
    # Return dictionary with multiple result columns
    return {
        'calculated_sum': param1 + param2,
        'calculated_product': param1 * param2,
        'calculated_ratio': param1 / param2 if param2 != 0 else 0
    }

def example_complex_expansion_function(row_data):
    """
    Example 4: Complex scenario - multiple rows with multiple columns
    """
    base_value = row_data.get('input_column', 0)
    category = row_data.get('category', 'default')
    
    # Generate different calculations based on category
    results = []
    
    if category == 'A':
        # Category A gets 2 rows
        results.append({
            'calculation_type': 'square',
            'result_value': base_value ** 2,
            'confidence': 0.95
        })
        results.append({
            'calculation_type': 'cube',
            'result_value': base_value ** 3,
            'confidence': 0.90
        })
    elif category == 'B':
        # Category B gets 3 rows
        for multiplier in [2, 4, 6]:
            results.append({
                'calculation_type': f'multiply_by_{multiplier}',
                'result_value': base_value * multiplier,
                'confidence': 0.85
            })
    else:
        # Default gets 1 row
        results.append({
            'calculation_type': 'identity',
            'result_value': base_value,
            'confidence': 1.0
        })
    
    return results

def example_api_call_function(row_data):
    """
    Example 5: Simulating API calls that return multiple records
    """
    user_id = row_data.get('user_id')
    
    # Simulate API response with multiple records per user
    # In real scenario, this would be an actual API call
    time.sleep(0.01)  # Simulate network delay
    
    # Simulate getting multiple transactions for this user
    transactions = []
    for i in range(2, 5):  # Random number of transactions
        transactions.append({
            'transaction_id': f"{user_id}_tx_{i}",
            'amount': i * 100,
            'transaction_type': 'credit' if i % 2 == 0 else 'debit',
            'timestamp': f"2024-01-{i:02d}"
        })
    
    return transactions

# =============================================================================
# SIMPLE USAGE EXAMPLES
# ========================================
