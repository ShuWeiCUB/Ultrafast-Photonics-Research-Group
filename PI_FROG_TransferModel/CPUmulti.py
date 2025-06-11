import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def compute_iteration(i):
    sum(x * x for x in range(1000000))  # Simulate CPU-heavy task
    return i

if __name__ == "__main__":  # Ensure proper multiprocessing in environments like Jupyter
    num_workers = multiprocessing.cpu_count()

    print(f"Number of workers: {num_workers}")
    iterations = 100  # Number of tasks
    
     # Test Sequential Execution (No multiprocessing)
    start_time = time.time()
    results_seq = [compute_iteration(i) for i in range(iterations)]
    print("Sequential execution time:", time.time() - start_time)

    # Test multiprocessing.Pool
    start_time = time.time()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results1 = pool.map(compute_iteration, range(iterations))
    print("multiprocessing.Pool time:", time.time() - start_time)
    
    # Test ProcessPoolExecutor
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results2 = list(executor.map(compute_iteration, range(iterations)))
    print("ProcessPoolExecutor time:", time.time() - start_time)
