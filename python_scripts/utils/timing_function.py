from time import perf_counter as timer


def time_function(func):
    def wrapper(*args, **kwargs):
        time1 = timer()
        output = func(*args, **kwargs)
        time2 = timer()
        print(f"Time to run '{func.__name__}': {time2-time1:.5f} seconds")
        return output

    return wrapper
