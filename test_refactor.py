
import sys
import numpy as np
import builtins
import io

# Mock input to simulate user interaction
input_sequence = [
    "3",       # Select Regression Tool
    "1,2;3,4", # Enter X
    "5;6",     # Enter y
    "0",       # Lambda
    "y",       # Visualize? (Should trigger visualize_regression)
    "7"        # Exit
]

def mock_input(prompt=""):
    print(prompt, end="")
    if input_sequence:
        val = input_sequence.pop(0)
        print(val)
        return val
    return ""

builtins.input = mock_input

# Redirect stdout to capture output
captured_output = io.StringIO()
sys.stdout = captured_output

try:
    import les_solver
    # Helper to check if visualize_regression handles arguments correctly
    # We call it directly to test argument passing
    X = np.array([[1], [2], [3]])
    y = np.array([[2], [4], [6]])
    w = np.array([[2]])
    
    # We need to mock plt.show to avoid blocking
    import matplotlib.pyplot as plt
    plt.show = lambda block=None: print("[Mock] plt.show() called")
    
    # Check if global function exists and runs
    print("Testing global visualize_regression...")
    # Mock input for the visualization prompt inside the function
    # The function asks "Do you want to see the plot? (y/n):"
    # So we need to push a 'y' into input_sequence for this call
    input_sequence.insert(0, 'y') 
    
    les_solver.visualize_regression(X, y, w, title="Test Plot")
    
    print("Test Passed!")
    
except Exception as e:
    print(f"Test Failed: {e}")
    import traceback
    traceback.print_exc()

sys.stdout = sys.__stdout__
print(captured_output.getvalue())
