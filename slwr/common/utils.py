import asyncio
from functools import wraps

def run_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Check if there's an active event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop is running
            loop = None

        if loop and loop.is_running():
            # If there's a running event loop, call it as an async function
            return func(*args, **kwargs)
        else:
            # If no event loop is running, create one and run the function
            return asyncio.run(func(*args, **kwargs))

    return wrapper
