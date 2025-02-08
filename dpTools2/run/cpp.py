import subprocess
import multiprocessing

def launch_cpp_script_with_triggers(executable, *args, triggering_substrings=None, functions=None, function_kwargs=None):
    """
    Launch the C++ executable in a subprocess while streaming its output.
    The function monitors the output for given triggering substrings and launches the
    corresponding function in a separate process when a trigger is found.

    Instead of keeping only the last 4 lines, this version accumulates the full output
    generated so far (i.e. the current state of the output text) and passes it to the
    triggered function via the keyword argument 'current_output'.
    
    Parameters:
        executable (str): Path to the C++ executable.
        *args: Additional arguments passed to the executable.
        triggering_substrings (list of str, optional): List of substrings to look for as triggers.
        functions (list of callable, optional): List of functions to call when a trigger is detected.
        function_kwargs (list of dict, optional): List of kwargs dictionaries for each respective function.
    """
    # Convert None values to empty lists.
    if triggering_substrings is None:
        triggering_substrings = []
    if functions is None:
        functions = []
    if function_kwargs is None:
        function_kwargs = []
        
    # If any triggers are provided, verify that all three lists have the same length.
    if triggering_substrings or functions or function_kwargs:
        if len(triggering_substrings) != len(functions) or len(triggering_substrings) != len(function_kwargs):
            raise ValueError("triggering_substrings, functions, and function_kwargs must have the same length.")
    
    # List to accumulate all output lines from the C++ executable.
    output_history = []
    
    process = subprocess.Popen(
        [executable] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr with stdout.
        bufsize=1,                   # Enable line buffering.
        universal_newlines=True      # Text-mode output.
    )
    print(f"Started {executable} with arguments {args}")
    
    # Read the output line-by-line as it is produced.
    for line in iter(process.stdout.readline, ''):
        print(line, end='')          # Print each line immediately.
        output_history.append(line)  # Accumulate the full output.
        
        # Check each trigger against the current line.
        for idx, trigger in enumerate(triggering_substrings):
            if trigger in line:
                print(f"\nTrigger '{trigger}' detected. Launching function: {functions[idx].__name__}")
                # Make a copy of the respective kwargs and insert the current full output.
                current_kwargs = dict(function_kwargs[idx])
                current_kwargs['current_output'] = list(output_history)
                # Launch the triggered function in a new process.
                p = multiprocessing.Process(target=functions[idx], kwargs=current_kwargs)
                p.start()

    process.stdout.close()
    process.wait()  # Wait for the process to complete.
    return process.returncode