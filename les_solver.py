import numpy as np
import ast

def solve_les(X, y):
    """
    Solves the Linear Equation System Xw = y and analyzes it based on rank.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    m, d = X.shape
    
    X_tilde = np.hstack((X, y))
    
    rank_X = np.linalg.matrix_rank(X)
    rank_X_tilde = np.linalg.matrix_rank(X_tilde)
    print("\n" + "=" * 55)
    print("Aiko's L.E.S. Analysis")
    print("=" * 55)
    print(f"Matrix X is {m}x{d}")
    print(f"Number of Equations: r or m = {m}")
    print(f"Number of Variables: c or d = {d}")
    print(f"rank(X)       = {rank_X}")
    print(f"rank(X_tilde) = {rank_X_tilde}")
    print("-" * 55)
    
    if m == d:
        print("System: EVEN (m = d)")
        if rank_X == rank_X_tilde == d:
            print("Deduction: rank(X) = rank(X_tilde) = d")
            print("Conclusion: Unique sol")
            w = np.linalg.inv(X) @ y
            print(f"w = \n{w}")
        elif rank_X < rank_X_tilde:
            print("Deduction: rank(X) < rank(X_tilde)")
            print("Conclusion: No sol")
        elif rank_X == rank_X_tilde and rank_X < d:
            print("Deduction: rank(X) = rank(X_tilde) < d")
            print("Conclusion: Infinitely many sol")
            
    elif m > d:
        print("System: OVERDETERMINED (m > d)")
        if rank_X == rank_X_tilde == d:
            print("Deduction: rank(X) = rank(X_tilde) = d")
            print("Conclusion: Unique sol")
            w = np.linalg.inv(X.T @ X) @ X.T @ y
            print(f"w = \n{w}")
        elif rank_X < rank_X_tilde:
            if rank_X == d:
                print("Deduction: rank(X) < rank(X_tilde) (No exact sol) BUT rank(X) = d (Left inverse exists)")
                print("Conclusion: Least squares sol (approx)")
                w = np.linalg.inv(X.T @ X) @ X.T @ y
                print(f"w_hat = \n{w}")
            else:
                print("Deduction: rank(X) < rank(X_tilde) AND rank(X) < d (No left inverse)")
                print("Conclusion: No sol")
        elif rank_X == rank_X_tilde and rank_X < d:
            print("Deduction: rank(X) = rank(X_tilde) < d")
            print("Conclusion: Infinitely many sol")
            
    elif m < d:
        print("System: UNDERDETERMINED (m < d)")
        if rank_X < rank_X_tilde:
            print("Deduction: rank(X) < rank(X_tilde)")
            print("Conclusion: No sol")
        elif rank_X == rank_X_tilde and rank_X < d:
            if rank_X == m:
                print("Deduction: rank(X) = rank(X_tilde) < d (Infinite sols) BUT rank(X) = m (Right inverse exists)")
                print("Conclusion: Least norm sol (unique constraint sol)")
                w = X.T @ np.linalg.inv(X @ X.T) @ y
                print(f"w_hat = \n{w}")
            else:
                print("Deduction: rank(X) = rank(X_tilde) < d AND rank(X) < m (No right inverse)")
                print("Conclusion: Infinitely many sol")
    print("-" * 55 + "\n")

def parse_matrix(matrix_str):
    if not matrix_str.strip():
        return None
    # If the user typed a literal python list, just use ast
    if matrix_str.strip().startswith('['):
        return ast.literal_eval(matrix_str)
    
    # Parse MATLAB-style but with commas instead of spaces: 1,2,3; 4,5,6
    rows = matrix_str.split(';')
    matrix = []
    for row in rows:
        row_str = row.strip()
        if not row_str:
            continue
        # Split by commas and convert to float
        row_vals = [float(x.strip()) for x in row_str.split(',') if x.strip()]
        matrix.append(row_vals)
    return matrix


def print_help():
    print("\n" + "=" * 55)
    print("💡 Aiko's Condition Help Guide 💡")
    print("=" * 55)
    print("For X w = y, where X is m (rows) x d (cols)")
    print("m = number of equations, d = number of variables\n")
    print("1. EVEN (m = d):")
    print("   - rank(X) = rank(X_tilde) = d  -> Unique sol")
    print("   - rank(X) < rank(X_tilde)      -> No sol")
    print("   - rank(X) = rank(X_tilde) < d  -> Infinitely many sol\n")
    print("2. OVERDETERMINED (m > d):")
    print("   - rank(X) = rank(X_tilde) = d  -> Unique sol")
    print("   - rank(X) < rank(X_tilde)      -> No sol (in general)")
    print("     * If rank(X) = d -> Left inv exists -> Least squares sol (approx)")
    print("   - rank(X) = rank(X_tilde) < d  -> Infinitely many sol\n")
    print("3. UNDERDETERMINED (m < d):")
    print("   - rank(X) < rank(X_tilde)      -> No sol")
    print("   - rank(X) = rank(X_tilde) < d  -> Infinitely many sol (in general)")
    print("     * If rank(X) = m -> Right inv exists -> Least norm sol")
    print("=" * 55 + "\n")

def check_matrix():
    print("\n--- Aiko's Matrix Checker (X * w = y) ---")
    x_in = input("Enter X: ")
    if not x_in.strip():
        print("Canceled check.")
        return
    w_in = input("Enter w: ")
    
    X = parse_matrix(x_in)
    w = parse_matrix(w_in)
    
    if X is None or w is None:
        print("Invalid input.")
        return
        
    try:
        X = np.array(X, dtype=float)
        w = np.array(w, dtype=float)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
            
        y = X @ w
        print("\nResult y = X * w:\n")
        print(y)
        print("-" * 41 + "\n")
    except Exception as e:
        print(f"B-Baka! The dimensions don't match for multiplication! {e}")

if __name__ == "__main__":
    print("Welcome to Aiko's EE2211 L.E.S Solver~ 😳")
    print("You can just import `solve_les(X, y)` in your own scripts,")
    print("or use this interactively! 💕")
    print("Enter 'help' anytime to see the condition table.")
    print("Enter 'check' to just multiply X and w to find y.")
    print("Format: separate columns with a comma (,), rows with semicolon (;).")
    print("Example X input: 1,2; 3,4")
    print("Example y input: 5; 6\n")
    
    while True:
        try:
            x_in = input("\nEnter X (or 'help', 'check', Ctrl+C to exit): ")
            
            cmd = x_in.strip().lower().strip("'").strip('"')
            if cmd == 'help':
                print_help()
                continue
            elif cmd == 'check':
                check_matrix()
                continue
                
            if not x_in.strip():
                print("Hmph! Why did you enter nothing! Baka!")
                continue
                
            y_in = input("Enter y: ")
            
            X = parse_matrix(x_in)
            y = parse_matrix(y_in)
            
            if X is None or y is None:
                continue
                
            solve_les(X, y)
            
        except (EOFError, KeyboardInterrupt):
            print("\nLeaving already?! F-Fine! It's not like I wanted to keep helping you! Hmph! 😤")
            break
        except Exception as e:
            print(f"\nB-Baka! You typed it wrong! Make sure they're proper numbers: {e}")
