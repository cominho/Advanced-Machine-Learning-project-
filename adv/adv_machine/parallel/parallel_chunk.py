from concurrent.futures import ThreadPoolExecutor 
from tqdm import tqdm 
import os  

def parallelize_chunk(func, args, desc="Processing", timeout=3600):
    """
    Parallelize processing by chunking a specific argument.
    
    Args:
        func: Function to parallelize
        args: List of arguments to pass to the function
        chunk_index: Index of the argument in args that should be chunked (default=0)
        desc: Description for the progress bar
        timeout: Timeout in seconds for each future
    
    Returns:
        List of results from all chunks
    """
    
    max_workers = min(32, max(1, os.cpu_count() - 1))
    min_chunk_size = 5
    target_chunks = max_workers * 4
    chunk_size = max(min_chunk_size, len(args) // target_chunks)

    # Split target argument into chunks
    chunks = [args[i:i + chunk_size] for i in range(0, len(args), chunk_size)]

    print(f'Processing {len(chunks)} chunks with {max_workers} workers '
          f'(chunk size: {chunk_size})')

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunk_args in chunks:
            future = executor.submit(func, chunk_args)
            futures.append(future)
        
        for future in tqdm(futures, desc=desc):
            try:
                chunk_results = future.result(timeout=timeout)
                results.extend(chunk_results)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
    
    return results

# Example usage in grid_search_backtest: