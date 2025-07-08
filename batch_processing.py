import pandas as pd
import multiprocessing as mp
from functools import partial
import tempfile
import os
import time
import logging
from pathlib import Path
from typing import Callable, Dict, Any, List
import json

class MemoryEfficientExcelProcessor:
    """
    Memory-efficient batch processor for large Excel files.
    Processes files in chunks to avoid memory overflow.
    """
    
    def __init__(self, input_file: str, output_file: str, 
                 chunk_size: int = 1000, num_processes: int = None):
        """
        Initialize the processor.
        
        Args:
            input_file: Path to input Excel file
            output_file: Path to output Excel file
            chunk_size: Number of rows to process per chunk
            num_processes: Number of parallel processes (default: CPU count - 1)
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.chunk_size = chunk_size
        self.num_processes = num_processes or mp.cpu_count() - 1
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_rows': 0,
            'successful_rows': 0,
            'failed_rows': 0,
            'chunks_processed': 0,
            'total_processing_time': 0
        }
    
    def process_row(self, row_data: Dict[str, Any], script_func: Callable) -> Dict[str, Any]:
        """
        Process a single row with error handling and timing.
        
        Args:
            row_data: Dictionary containing row data
            script_func: Function to process the row
            
        Returns:
            Updated row dictionary with results
        """
        start_time = time.time()
        
        try:
            # Extract input data (modify based on your column names)
            input_data = row_data.get('input_column')
            
            # You can also extract multiple columns:
            # input_data = {
            #     'param1': row_data.get('param1'),
            #     'param2': row_data.get('param2')
            # }
            
            # Process using your function
            result = script_func(input_data)
            
            # Update row with results
            row_data['result'] = result
            row_data['status'] = 'success'
            row_data['error'] = None
            row_data['processing_time'] = time.time() - start_time
            
        except Exception as e:
            # Handle errors gracefully
            row_data['result'] = None
            row_data['status'] = 'failed'
            row_data['error'] = str(e)
            row_data['processing_time'] = time.time() - start_time
            
            # Log error (optional)
            self.logger.warning(f"Error processing row: {e}")
        
        return row_data
    
    def process_chunk_parallel(self, chunk_df: pd.DataFrame, 
                             script_func: Callable) -> pd.DataFrame:
        """
        Process a chunk of data in parallel.
        
        Args:
            chunk_df: DataFrame chunk to process
            script_func: Function to process each row
            
        Returns:
            Processed DataFrame chunk
        """
        # Convert DataFrame to list of dictionaries
        rows = chunk_df.to_dict('records')
        
        # Create partial function for multiprocessing
        process_func = partial(self.process_row, script_func=script_func)
        
        # Process rows in parallel
        with mp.Pool(processes=self.num_processes) as pool:
            processed_rows = pool.map(process_func, rows)
        
        return pd.DataFrame(processed_rows)
    
    def get_total_rows(self) -> int:
        """Get total number of rows in the input file for progress tracking."""
        try:
            # Quick way to get row count without loading all data
            df_sample = pd.read_excel(self.input_file, nrows=0)  # Just headers
            
            # For very large files, this is more memory efficient
            total_rows = 0
            for chunk in pd.read_excel(self.input_file, chunksize=self.chunk_size):
                total_rows += len(chunk)
            
            return total_rows
        except Exception as e:
            self.logger.warning(f"Could not determine total rows: {e}")
            return 0
    
    def process_excel_chunks(self, script_func: Callable) -> bool:
        """
        Main processing method that handles the entire workflow.
        
        Args:
            script_func: Function to process each row
            
        Returns:
            True if processing completed successfully
        """
        start_time = time.time()
        temp_files = []
        
        try:
            # Get total rows for progress tracking
            total_rows = self.get_total_rows()
            self.logger.info(f"Starting processing of {total_rows} rows in chunks of {self.chunk_size}")
            
            # Process each chunk
            chunk_count = 0
            processed_rows = 0
            
            for chunk_df in pd.read_excel(self.input_file, chunksize=self.chunk_size):
                chunk_start_time = time.time()
                chunk_count += 1
                
                self.logger.info(f"Processing chunk {chunk_count} ({len(chunk_df)} rows)...")
                
                # Add result columns if they don't exist
                required_columns = ['result', 'status', 'error', 'processing_time']
                for col in required_columns:
                    if col not in chunk_df.columns:
                        chunk_df[col] = None
                
                # Process chunk in parallel
                processed_chunk = self.process_chunk_parallel(chunk_df, script_func)
                
                # Update statistics
                chunk_successful = (processed_chunk['status'] == 'success').sum()
                chunk_failed = (processed_chunk['status'] == 'failed').sum()
                
                self.stats['successful_rows'] += chunk_successful
                self.stats['failed_rows'] += chunk_failed
                self.stats['chunks_processed'] += 1
                
                processed_rows += len(processed_chunk)
                
                # Save chunk to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                processed_chunk.to_excel(temp_file.name, index=False)
                temp_files.append(temp_file.name)
                temp_file.close()
                
                chunk_time = time.time() - chunk_start_time
                
                # Progress update
                progress = (processed_rows / total_rows * 100) if total_rows > 0 else 0
                self.logger.info(f"Chunk {chunk_count} completed in {chunk_time:.2f}s. "
                               f"Progress: {progress:.1f}% ({processed_rows}/{total_rows})")
                self.logger.info(f"Chunk results: {chunk_successful} successful, {chunk_failed} failed")
            
            # Combine all chunks
            self.logger.info("Combining all chunks into final result...")
            self._combine_chunks(temp_files)
            
            # Update final statistics
            self.stats['total_rows'] = processed_rows
            self.stats['total_processing_time'] = time.time() - start_time
            
            # Generate and save summary
            self._generate_summary()
            
            self.logger.info(f"Processing complete! Results saved to {self.output_file}")
            self.logger.info(f"Total time: {self.stats['total_processing_time']:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return False
            
        finally:
            # Always cleanup temporary files
            self._cleanup_temp_files(temp_files)
    
    def _combine_chunks(self, temp_files: List[str]) -> None:
        """Combine all temporary chunk files into final output."""
        if not temp_files:
            raise ValueError("No temporary files to combine")
        
        # Read and combine all chunks
        chunk_dataframes = []
        for temp_file in temp_files:
            chunk_df = pd.read_excel(temp_file)
            chunk_dataframes.append(chunk_df)
        
        # Combine all chunks
        combined_df = pd.concat(chunk_dataframes, ignore_index=True)
        
        # Save final result
        combined_df.to_excel(self.output_file, index=False)
        
        self.logger.info(f"Combined {len(chunk_dataframes)} chunks into {len(combined_df)} total rows")
    
    def _cleanup_temp_files(self, temp_files: List[str]) -> None:
        """Clean up temporary files."""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                self.logger.warning(f"Could not delete temp file {temp_file}: {e}")
    
    def _generate_summary(self) -> None:
        """Generate and save processing summary."""
        summary = {
            'input_file': str(self.input_file),
            'output_file': str(self.output_file),
            'processing_parameters': {
                'chunk_size': self.chunk_size,
                'num_processes': self.num_processes
            },
            'results': {
                'total_rows': self.stats['total_rows'],
                'successful_rows': self.stats['successful_rows'],
                'failed_rows': self.stats['failed_rows'],
                'success_rate': (self.stats['successful_rows'] / self.stats['total_rows'] * 100) 
                               if self.stats['total_rows'] > 0 else 0,
                'chunks_processed': self.stats['chunks_processed']
            },
            'timing': {
                'total_processing_time': self.stats['total_processing_time'],
                'average_time_per_chunk': (self.stats['total_processing_time'] / self.stats['chunks_processed'])
                                         if self.stats['chunks_processed'] > 0 else 0,
                'rows_per_second': (self.stats['total_rows'] / self.stats['total_processing_time'])
                                  if self.stats['total_processing_time'] > 0 else 0
            }
        }
        
        # Save summary to JSON file
        summary_file = self.output_file.with_suffix('.summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to {summary_file}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total rows processed: {summary['results']['total_rows']}")
        print(f"Successful: {summary['results']['successful_rows']}")
        print(f"Failed: {summary['results']['failed_rows']}")
        print(f"Success rate: {summary['results']['success_rate']:.2f}%")
        print(f"Total time: {summary['timing']['total_processing_time']:.2f} seconds")
        print(f"Speed: {summary['timing']['rows_per_second']:.2f} rows/second")
        print("="*50)

# Example usage
def example_script_function(input_data):
    """
    Example processing function - replace with your actual logic.
    
    Args:
        input_data: Input value from Excel row
        
    Returns:
        Processed result
    """
    # Simulate some processing time
    time.sleep(0.01)  # Remove this in real implementation
    
    # Your actual processing logic here
    if input_data is None:
        raise ValueError("Input data is None")
    
    # Example: square the input
    result = input_data ** 2
    
    return result

# Example for multiple input columns
def example_multi_input_function(input_data):
    """
    Example function that handles multiple input columns.
    
    Args:
        input_data: Dictionary with multiple input values
        
    Returns:
        Processed result
    """
    param1 = input_data.get('param1', 0)
    param2 = input_data.get('param2', 0)
    
    # Your processing logic
    result = param1 * param2 + 100
    
    return result

if __name__ == "__main__":
    # Example usage
    processor = MemoryEfficientExcelProcessor(
        input_file="large_input_data.xlsx",
        output_file="processed_results.xlsx",
        chunk_size=500,  # Process 500 rows at a time
        num_processes=4  # Use 4 parallel processes
    )
    
    # Run the processing
    success = processor.process_excel_chunks(example_script_function)
    
    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed. Check the logs for details.")
