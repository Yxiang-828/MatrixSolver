import numpy as np
import ast
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from colorama import init, Fore, Style

# Initialize colorama for Windows ANSI support
init(autoreset=True)

# ─── Color Palette ─────────────────────────────────────
# Normal text  : default terminal (white)
# User input   : CYAN   — prompts asking you to type something
# Result output: GREEN  — computed answers, weights, scores
# Error/warn   : YELLOW — warnings/errors
# ────────────────────────────────────────────────────────
C_INPUT  = Fore.CYAN                    # input prompt colour
C_RESULT = Fore.GREEN + Style.BRIGHT    # computed result colour
C_WARN   = Fore.YELLOW                  # warnings / errors
C_RESET  = Style.RESET_ALL             # back to normal

def cinput(prompt=""):
    """Cyan-coloured input prompt."""
    return input(f"{C_INPUT}{prompt}{C_RESET}")

def rprint(*args, **kwargs):
    """Green-coloured print for results."""
    msg = " ".join(str(a) for a in args)
    print(f"{C_RESULT}{msg}{C_RESET}", **kwargs)

def wprint(*args, **kwargs):
    """Yellow-coloured print for warnings/errors."""
    msg = " ".join(str(a) for a in args)
    print(f"{C_WARN}{msg}{C_RESET}", **kwargs)

# Helper functions for combinatorics
def nPr(n, r):
    try:
        if r > n: return 0
        return math.factorial(n) // math.factorial(n - r)
    except ValueError:
        return None

def nCr(n, r):
    try:
        if r > n: return 0
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
    except ValueError:
        return None


def mprint(arr, label=""):
    """Print a numpy array in green, then show the MATLAB-style copy-paste format."""
    arr = np.atleast_2d(arr)
    # Green pretty-print
    if label:
        print(label)
    rprint(arr)
    # Build MATLAB-style string: round to 4dp for readability
    rows_str = "; ".join(
        ", ".join(f"{v:.4f}" for v in row)
        for row in arr
    )
    print(f"{Fore.CYAN}  📋 paste-ready: {rows_str}{C_RESET}")
    
def visualize_regression(X_data, y_data, w_model, title="Linear Regression"):
    """Plot regression results: Line for 1D, Actual vs Pred for >1D."""
    try:
        print("\n📈 [Visualisation Check]")
        user_choice = cinput("Do you want to see the plot? (y/n): ").strip().lower()
        if user_choice not in ['y', 'yes', '1']:
            return

        print("Generating plot... (Check popup window)")
        m, d = X_data.shape
        y_pred = X_data @ w_model
        
        # Ensure 1D arrays
        y_flat = y_data.flatten()
        pred_flat = y_pred.flatten()

        # Calculate Metrics (R2, MSE) for annotation
        ss_res = np.sum((y_flat - pred_flat) ** 2)
        ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        mse = np.mean((y_flat - pred_flat) ** 2)

        plt.figure(figsize=(10, 6))
        
        # Check 1D & Bias
        is_1d_reg = False
        feature_col = None
        has_bias = False
        bias_col_idx = -1
        
        # Check columns for bias (all 1s)
        for c in range(d):
            if np.allclose(X_data[:, c], 1):
                has_bias = True
                bias_col_idx = c
                break

        if d == 1:
            is_1d_reg = True
            feature_col = X_data[:, 0]
            xlab = "Feature X"
        elif d == 2 and has_bias:
            is_1d_reg = True
            feat_idx = 1 - bias_col_idx
            feature_col = X_data[:, feat_idx]
            xlab = f"Feature X (col {feat_idx})"
        
        # Subtitle & Stats Text
        subtitle = "Bias Term Included" if has_bias else "No Bias Term (origin-constrained)"
        stats_text = f"$R^2 = {r2:.4f}$\nMSE$ = {mse:.4f}$"
        
        # Stats box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        if is_1d_reg and feature_col is not None:
            # --- 1D Feature Plot ---
            # Jitter X for visibility
            x_range = np.max(feature_col) - np.min(feature_col) 
            if x_range == 0: x_range = 1
            x_jitter = feature_col + np.random.normal(0, 0.015 * x_range, size=len(feature_col))
            
            plt.scatter(x_jitter, y_flat, color='blue', label='Actual (Jittered)', alpha=0.6, s=40, edgecolors='k')
            
            # Sort for clean regression line
            sort_idx = np.argsort(feature_col)
            plt.plot(feature_col[sort_idx], pred_flat[sort_idx], color='red', linewidth=2, label='Regression Line')
            
            # Residual Lines (Vertical from point to PREDICTED y)
            for i in range(m):
                # Line from (x[i], y[i]) to (x[i], pred[i])
                plt.plot([feature_col[i], feature_col[i]], [y_flat[i], pred_flat[i]], 'k:', alpha=0.2)
                # Annotate point with (x, y) value
                label = f"({feature_col[i]:.2f}, {y_flat[i]:.2f})"
                plt.annotate(label, (x_jitter[i], y_flat[i]), xytext=(5, 5), 
                             textcoords='offset points', fontsize=8, alpha=0.9)
            
            if np.min(y_flat) >= 0 and np.min(pred_flat) >= -0.1: plt.ylim(bottom=0)
            if np.min(feature_col) >= 0: plt.xlim(left=0)

            plt.title(f"{title} (1D Fit)\n{subtitle}")
            plt.xlabel(xlab)
            plt.ylabel("Target y")
            plt.legend()
            
        else:
            # --- Actual vs Predicted Plot ---
            plt.scatter(y_flat, pred_flat, color='purple', label='Predictions', alpha=0.6, s=40, edgecolors='k')
            
            # Perfect Fit Line y=x
            all_vals = np.concatenate([y_flat, pred_flat])
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            buff = 0.05 * (max_val - min_val) if max_val != min_val else 1.0
            plt.plot([min_val - buff, max_val + buff], [min_val - buff, max_val + buff], 'r--', label='Perfect Fit (y=x)')
            
            # Residual Lines (Vertical distance to y=x imply error in prediction vs actual)
            for i in range(m):
                plt.plot([y_flat[i], y_flat[i]], [pred_flat[i], y_flat[i]], 'k:', alpha=0.2)
                # Annotate point with (Actual, Pred)
                label = f"({y_flat[i]:.2f}, {pred_flat[i]:.2f})"
                plt.annotate(label, (y_flat[i], pred_flat[i]), xytext=(5, 5), 
                             textcoords='offset points', fontsize=8, alpha=0.9)

            if np.min(y_flat) >= 0 and np.min(pred_flat) >= -0.1:
                plt.xlim(left=0)
                plt.ylim(bottom=0)

            plt.title(f"{title}: Actual vs Predicted\n{subtitle}")
            plt.xlabel("Actual y")
            plt.ylabel("Predicted y")
            plt.legend()

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show() 
        print(f"{Fore.GREEN}  [Graph closed]{C_RESET}")

    except Exception as e:
        wprint(f"[!] Plotting failed. Is matplotlib installed? Error: {e}")

def visualize_classification(X, Y, W, poly=None, title="Classification Boundary"):
    """
    Visualizes classification results.
    - If 2D features (raw): Plots data points colored by class and decision boundary.
    - Only supports visualization for 2 input features currently.
    """
    try:
        do_plot = cinput("Visualise decision boundary (2D)? (y/N): ").lower()
        if do_plot not in ['y', 'yes', '1']: return

        # Check dimensions
        # X should be (m, d_features). If d_features != 2 (excl bias) it's hard to plot.
        # We need the RAW features before polynomial expansion for the meshgrid.
        # However, the W is trained on expanded features.
        
        # Heuristic: Check if X has 2 columns (or 3 with bias).
        m, d = X.shape
        feature_cols = []
        
        # Identify non-bias columns
        for i in range(d):
            if not np.allclose(X[:, i], 1):
                feature_cols.append(i)
        
        if len(feature_cols) != 2:
            wprint(f"[!] Can only visualise 2D features. Found {len(feature_cols)} features.")
            return
            
        f1_idx, f2_idx = feature_cols
        X_raw = X[:, [f1_idx, f2_idx]]
        
        # Convert One-Hot Y to class indices for colouring
        if Y.ndim > 1 and Y.shape[1] > 1:
            y_classes = np.argmax(Y, axis=1)
        else:
            y_classes = Y.flatten()
            
        # Create Meshgrid
        x_min, x_max = X_raw[:, 0].min() - 1, X_raw[:, 0].max() + 1
        y_min, y_max = X_raw[:, 1].min() - 1, X_raw[:, 1].max() + 1
        h = 0.05 # step size
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Flatten meshgrid points
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Prepare mesh points for prediction (Add bias / Poly features)
        # Note: We assume the model W corresponds to the structure of X passed in.
        # If X was poly-transformed, we need to poly-transform the mesh points.
        # If X was just bias-added, we add bias.
        
        # We need to know how to transform mesh points to match W's expectations.
        # This is tricky without passing 'transform_func'.
        # Simplified: We assume X passed here is the TRAINING X (already transformed).
        # Wait, if X is already transformed (e.g. poly degree 2), it has many cols.
        # If we plotted raw X, we can't easily multiply by W unless we repeat the transform.
        
        wprint("[i] Visualisation needs to map 2D mesh -> Model features.")
        wprint("    This is complex if polynomial features were used.")
        wprint("    Plotting just the training points...")

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_raw[:, 0], X_raw[:, 1], c=y_classes, cmap='viridis', s=50, edgecolors='k')
        plt.colorbar(scatter, label='Class')
        plt.xlabel(f"Feature {f1_idx}")
        plt.ylabel(f"Feature {f2_idx}")
        plt.title(title + " (Training Data)")
        plt.grid(True, alpha=0.3)
        plt.show()
        print(f"{Fore.GREEN}  [Graph window opened. Close it to continue...]{C_RESET}")

    except Exception as e:
        wprint(f"[!] Visualisation failed: {e}")

def add_bias(X):
    return np.insert(X, 0, 1, axis=1)

def print_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rprint(f"MSE: {mse}")

def parse_matrix(matrix_str, name="Matrix"):
    """
    Parses MATLAB-style string inputs into a numpy array.
    """
    if str(matrix_str).strip() == '':
        return None
        
    try:
        # If the user typed a literal python list, just use ast
        matrix_str_clean = matrix_str.strip()
        if matrix_str_clean.startswith('['):
            return np.array(ast.literal_eval(matrix_str_clean), dtype=float)
        
        # Parse MATLAB-style: 1,2,3; 4,5,6
        rows = matrix_str_clean.split(';')
        matrix = []
        for row in rows:
            row_str = row.strip()
            if not row_str:
                continue
            row_vals = [float(x.strip()) for x in row_str.split(',') if x.strip()]
            matrix.append(row_vals)
        return np.array(matrix, dtype=float)
    except Exception as e:
        wprint(f"\n[!] B-Baka! You typed {name} wrong! Error: {e}")
        return None

# ==========================================
# TOOL 1: MATRIX MATH
# ==========================================
def tool_matrix_math():
    while True:
        print("\n--- Aiko's Matrix Math & Checker ---")
        print("1) Matrix Multiplication (X * W = y)")
        print("2) Dot Product (v1 . v2)")
        print("3) Transpose (X^T)")
        print("B) Back to Main Menu")
        sub_choice = cinput("Select sub-tool (1-3 or B): ").strip().upper()

        if sub_choice == 'B':
            break

        if sub_choice == '1':
            X = parse_matrix(cinput("Enter X: "), "X")
            if X is None: continue
            bias_in = cinput("Add Bias column of 1s to X? (y/N): ").strip().lower()
            if bias_in in ['y', 'yes']:
                X = add_bias(X)
                print(f"(Bias added → X is now {X.shape[0]}×{X.shape[1]})")
            w = parse_matrix(cinput("Enter w: "), "w")
            if w is None: continue
            try:
                if w.ndim == 1: w = w.reshape(-1, 1)
                y = X @ w
                print("\nResult y = X * w:")
                mprint(y)
                print("-" * 41)
            except Exception as e:
                wprint(f"[!] B-Baka! The dimensions don't match for multiplication! {e}")
        elif sub_choice == '2':
            v1 = parse_matrix(cinput("Enter v1 (row or col): "), "v1")
            if v1 is None: continue
            v2 = parse_matrix(cinput("Enter v2 (row or col): "), "v2")
            if v2 is None: continue
            try:
                v1_flat = v1.flatten()
                v2_flat = v2.flatten()
                if len(v1_flat) != len(v2_flat):
                    wprint(f"[!] B-Baka! Vectors must be the same length!")
                    continue
                dot_prod = np.dot(v1_flat, v2_flat)
                print("\nResult v1 . v2:")
                rprint(dot_prod)
                print("-" * 41)
            except Exception as e:
                wprint(f"[!] Error: {e}")
        elif sub_choice == '3':
            X = parse_matrix(cinput("Enter X: "), "X")
            if X is None: continue
            try:
                X_T = X.T
                print("\nResult Transpose X^T:")
                mprint(X_T)
                print("-" * 41)
            except Exception as e:
                wprint(f"[!] Error: {e}")
        else:
            wprint("[!] Invalid sub-tool choice.")

# ==========================================
# TOOL 2: BULK MATRIX ANALYSER
# ==========================================
def tool_det_inverse():
    while True:
        print("\n--- Aiko's Bulk Matrix Analyser ---")
        print("B) Back to Main Menu")
        print("⚠️  Input: any matrix X (m rows × d cols). Commas & semicolons.")
        
        val = cinput("Enter matrix X (or B to back): ")
        if val.strip().upper() == 'B':
            break
            
        X = parse_matrix(val, "X")
        if X is None: continue

        m, d = X.shape
        r = np.linalg.matrix_rank(X)
        cond = np.linalg.cond(X)
        SEP = "=" * 54
    
        print(f"\n{SEP}")
        print("📊 BULK MATRIX ANALYSIS")
        print(SEP)
    
        # --- Shape & Rank ---
        print(f"\n📐 SHAPE & RANK")
        print(f"  Shape        : {m} rows (m) × {d} cols (d)")
        rprint(f"  rank(X)      = {r}")
        rprint(f"  max rank     = min(m,d) = min({m},{d}) = {min(m,d)}")
        if r == min(m, d):
            rprint(f"  Full Rank    ✓ (rank = min(m,d))")
        else:
            wprint(f"  RANK DEFICIENT — missing {min(m,d)-r} dimension(s)")
    
        # --- System Classification ---
        print(f"\n⚙️  SYSTEM TYPE  (m={m} vs d={d})")
        if m > d:
            print(f"  OVERDETERMINED (m > d): more equations than unknowns")
            print(f"  → Exact solution unlikely. Least-squares approximation used.")
        elif m == d:
            print(f"  EVEN (m = d): same equations as unknowns")
            print(f"  → Exact solution possible IF rank(X) = d.")
        else:
            wprint(f"  UNDERDETERMINED (m < d): fewer equations than unknowns")
            wprint(f"  → Infinite solutions possible. Minimum-norm solution used.")
    
        # --- Inverse Availability ---
        print(f"\n🔄 INVERSE AVAILABILITY")
        # Standard inverse: only square + full rank
        if m == d:
            if r == d:
                rprint(f"  ✅ Standard Inverse (X⁻¹) : YES — square & full rank")
                rprint(f"     Formula : X⁻¹ directly")
            else:
                wprint(f"  ❌ Standard Inverse (X⁻¹) : NO — square but RANK DEFICIENT (rank={r} < d={d})")
        else:
            print(f"  —  Standard Inverse (X⁻¹) : N/A — not square (m≠d)")
    
        # Left inverse: m >= d and rank(X) == d
        if r == d:
            rprint(f"  ✅ Left Inverse  (XᵀX)⁻¹Xᵀ : YES — rank(X) = d = {d}")
            rprint(f"     Formula : (XᵀX)⁻¹ Xᵀ     | Used for: Least Squares / OLS")
        else:
            wprint(f"  ❌ Left Inverse  (XᵀX)⁻¹Xᵀ : NO  — rank(X)={r} < d={d}")
    
        # Right inverse: m <= d and rank(X) == m
        if r == m:
            rprint(f"  ✅ Right Inverse Xᵀ(XXᵀ)⁻¹  : YES — rank(X) = m = {m}")
            rprint(f"     Formula : Xᵀ(XXᵀ)⁻¹       | Used for: Least Norm / Underdetermined")
        else:
            wprint(f"  ❌ Right Inverse Xᵀ(XXᵀ)⁻¹  : NO  — rank(X)={r} < m={m}")
    
        # --- Ridge Forms ---
        print(f"\n🔵 RIDGE REGRESSION FORMS  (both always computable)")
        if m >= d:
            rprint(f"  ✅ PRIMAL preferred (m≥d={d}): W = (XᵀX + λI_d)⁻¹ Xᵀ y   [d×d inversion]")
            print(f"     DUAL   available (m<d):  W = Xᵀ(XXᵀ + λI_m)⁻¹ y    [m×m inversion]")
        else:
            print(f"     PRIMAL available (m≥d):  W = (XᵀX + λI_d)⁻¹ Xᵀ y   [d×d inversion]")
            rprint(f"  ✅ DUAL   preferred (m<d={m}):  W = Xᵀ(XXᵀ + λI_m)⁻¹ y    [m×m inversion, smaller]")
    
        # --- Determinant (square only) ---
        if m == d:
            print(f"\n🔢 DETERMINANT  (square matrix)")
            det_X = np.linalg.det(X)
            rprint(f"  det(X) = {det_X:.6f}")
            if np.isclose(det_X, 0):
                wprint(f"  → det ≈ 0 : SINGULAR — columns linearly dependent, no standard inverse")
            else:
                print(f"  → det ≠ 0 : INVERTIBLE")
            print(f"  (Computing inverse...)")
            if not np.isclose(det_X, 0):
                inv_X = np.linalg.inv(X)
                print(f"  X⁻¹ =")
                mprint(inv_X)
    
        # --- Condition Number ---
        print(f"\n📈 CONDITION NUMBER")
        if cond > 1e10:
            wprint(f"  cond(X) = {cond:.3e}  ⚠️  SEVERELY ILL-CONDITIONED — numerically unstable!")
        elif cond > 1e6:
            wprint(f"  cond(X) = {cond:.3e}  ⚠️  ILL-CONDITIONED — Ridge regularisation strongly recommended")
        elif cond > 1e3:
            print(f"  cond(X) = {cond:.3e}  ⚠️  Mildly ill-conditioned — Ridge may help")
        else:
            rprint(f"  cond(X) = {cond:.3e}  ✓  Well-conditioned")
    
        print(f"\n{SEP}")
    
        # --- Optional: Augmented [X|y] ---
        aug = cinput("\nAlso analyse augmented [X|y] for L.E.S.? (y/N): ").strip().lower()
        if aug in ['y', 'yes']:
            print("⚠️  y must have same number of rows as X.")
            y_aug = parse_matrix(cinput("Enter y (same number of rows as X): "), "y")
            if y_aug is None: return
            if y_aug.ndim == 1: y_aug = y_aug.reshape(-1, 1)
            if y_aug.shape[0] != m:
                wprint(f"[!] y has {y_aug.shape[0]} rows but X has {m}. Must match!"); continue
            X_tilde = np.hstack((X, y_aug))
            r_tilde = np.linalg.matrix_rank(X_tilde)
            print(f"\n{'='*54}")
            print(f"📋 L.E.S. ANALYSIS  for Xw = y")
            print(f"{'='*54}")
            rprint(f"  rank(X)       = {r}")
            rprint(f"  rank([X|y])   = {r_tilde}")
            print(f"  d (variables) = {d}")
            print(f"  m (equations) = {m}")
            print()
            if r < r_tilde:
                wprint(f"  rank(X) < rank([X|y])  →  NO SOLUTION (inconsistent)")
                wprint(f"  Reason: y is not in the column space of X.")
            elif r == r_tilde == d:
                if m == d:
                    rprint(f"  rank(X) = rank([X|y]) = d = m  →  UNIQUE SOLUTION")
                    rprint(f"  Use: w = X⁻¹y")
                else:
                    rprint(f"  rank(X) = rank([X|y]) = d < m  →  UNIQUE SOLUTION (overdetermined consistent)")
                    rprint(f"  Use: w = (XᵀX)⁻¹Xᵀy  (Left inverse / OLS)")
            elif r == r_tilde < d:
                wprint(f"  rank(X) = rank([X|y]) = {r} < d={d}  →  INFINITE SOLUTIONS")
                if r == m:
                    rprint(f"  rank(X) = m = {m}  →  Right inverse exists")
                    rprint(f"  Use: ŵ = Xᵀ(XXᵀ)⁻¹y  (Minimum norm / Least norm solution)")
                else:
                    wprint(f"  rank(X) < m  →  No right inverse. General infinite family of solutions.")
            print(f"{'='*54}\n")
    
    
    
    # ==========================================
    # TOOL 3: LINEAR REGRESSION / L.E.S. SOLVER
    # ==========================================
def tool_solve_les():
    while True:
        print("\n--- Aiko's L.E.S. & Ridge Regression Solver ---")
        print("B) Back to Main Menu")
        val = cinput("Enter X (or B to back): ")
        if val.strip().upper() == 'B':
            break

        X = parse_matrix(val, "X")
        if X is None: continue
        
        val_y = cinput("Enter y: ")
        if val_y.strip().upper() == 'B': break
        y = parse_matrix(val_y, "y")
        if y is None: continue
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        lam_in = cinput("Enter Lambda for Ridge (default 0): ").strip()
        bias_in = cinput("Add Bias column of 1s? (y/N): ").strip().lower()
        
        lamda = float(lam_in) if lam_in else 0.0
        apply_bias = True if bias_in in ['y', 'yes'] else False
    
        if apply_bias:
            X = add_bias(X)
            
        m, d = X.shape
        X_tilde = np.hstack((X, y))
        
        rank_X = np.linalg.matrix_rank(X)
        rank_X_tilde = np.linalg.matrix_rank(X_tilde)
        
        print("\n" + "=" * 55)
        print("Aiko's Analysis Results")
        print("=" * 55)
        print(f"Matrix X is {m}x{d}")
        if apply_bias:
            print(f"(Bias column of 1s was automatically added to X!)")
        print(f"Number of Equations: r or n = {m}")
        print(f"Number of Variables: c or d = {d}")
        print(f"rank(X)       = {rank_X}")
        print(f"rank(X_tilde) = {rank_X_tilde}")
        print(f"Lambda (L2)   = {lamda}")
        print("-" * 55)
        
        # Visualization delegated to global function visualize_regression
    
        # RIDGE REGRESSION
        if lamda > 0:
            if d <= m:
                print("System: OVERDETERMINED (m >= d) with Lambda > 0")
                print("Action: Ridge regression (PRIMAL form)")
                I = np.eye(d)
                w = np.linalg.inv(X.T @ X + lamda * I) @ X.T @ y
                print("w ="); mprint(w)
                print_mse(y, X @ w)
                visualize_regression(X, y, w, title=f"Ridge Regression (Primal, L2={lamda})")
            else:
                print("System: UNDERDETERMINED (m < d) with Lambda > 0")
                print("Action: Ridge regression (DUAL form)")
                I = np.eye(m)
                w = X.T @ np.linalg.inv(X @ X.T + lamda * I) @ y
                print("w ="); mprint(w)
                print_mse(y, X @ w)
                visualize_regression(X, y, w, title=f"Ridge Regression (Dual, L2={lamda})")
            print("-" * 55 + "\n")
            continue # Continue loop instead of return
        
        # EXACT & LEAST SQUARES & LEAST NORM
        if m == d:
            print("System: EVEN (m = d)")
            if rank_X == rank_X_tilde == d:
                print("Deduction: rank(X) = rank(X_tilde) = d")
                print("Conclusion: Unique sol")
                w = np.linalg.inv(X) @ y
                print("w ="); mprint(w)
                print_mse(y, X @ w)
                visualize_regression(X, y, w, title="Exact Solution Fit")
            elif rank_X < rank_X_tilde:
                print("Deduction: rank(X) < rank(X_tilde) (No exact sol)")
                print("Conclusion: Least squares sol (approx) via Pseudo-inverse")
                w = np.linalg.pinv(X) @ y
                print("w_hat (OLS) ="); mprint(w)
                print_mse(y, X @ w)
                visualize_regression(X, y, w, title="OLS (Approx) Fit")
            elif rank_X == rank_X_tilde and rank_X < d:
                print("Deduction: rank(X) = rank(X_tilde) < d")
                print("Conclusion: Infinitely many sol")
                
        elif m > d:
            print("System: OVERDETERMINED (m > d)")
            if rank_X == rank_X_tilde == d:
                print("Deduction: rank(X) = rank(X_tilde) = d")
                print("Conclusion: Unique sol")
                w = np.linalg.inv(X.T @ X) @ X.T @ y
                print("w ="); mprint(w)
                print_mse(y, X @ w)
                visualize_regression(X, y, w, title="OLS Unique Fit")
            elif rank_X < rank_X_tilde:
                if rank_X == d:
                    print("Deduction: rank(X) < rank(X_tilde) BUT rank(X) = d (Left inv exists)")
                    print("Conclusion: Least squares sol (approx)")
                    w = np.linalg.inv(X.T @ X) @ X.T @ y
                    print("w_hat ="); mprint(w)
                    print_mse(y, X @ w)
                    visualize_regression(X, y, w, title="OLS (Left Inverse) Fit")
                else:
                    wprint("Deduction: rank(X) < rank(X_tilde) AND rank(X) < d (No left inverse)")
                    wprint("Conclusion: No sol")
            elif rank_X == rank_X_tilde and rank_X < d:
                print("Deduction: rank(X) = rank(X_tilde) < d")
                print("Conclusion: Infinitely many sol")
                
        elif m < d:
            print("System: UNDERDETERMINED (m < d)")
            if rank_X < rank_X_tilde:
                wprint("Deduction: rank(X) < rank(X_tilde)")
                wprint("Conclusion: No sol")
            elif rank_X == rank_X_tilde and rank_X < d:
                if rank_X == m:
                    print("Deduction: rank(X) = rank(X_tilde) < d (Infinite sols) BUT rank(X) = m (Right inv exists)")
                    print("Conclusion: Least norm sol")
                    w = X.T @ np.linalg.inv(X @ X.T) @ y
                    print("w_hat ="); mprint(w)
                    print_mse(y, X @ w)
                    visualize_regression(X, y, w, title="Minimum Norm Solution")
                else:
                    wprint("Deduction: rank(X) = rank(X_tilde) < d AND rank(X) < m")
                    wprint("Conclusion: Infinitely many sol")
        print("-" * 55 + "\n")
    
    # ==========================================
    # TOOL 4: CLASSIFICATION & POLYNOMIAL
    # ==========================================
def tool_classification_poly():
    while True:
        print("\n--- Aiko's Classification & Polynomial Tool ---")
        print("1) Binary & Multi-class Predictor  (you supply pre-trained W)")
        print("2) Polynomial Features Expansion   (expand X, no training)")
        print("3) Multi-class OLS Train + Predict (train W from X & Y, then classify)")
        print("B) Back to Main Menu")
        sub_choice = cinput("Select sub-tool (1-3 or B): ").strip().upper()
        
        if sub_choice == 'B':
            break

        if sub_choice == '1':
            print("Note: For Binary, W is a column vector. For Multi-class, W is a matrix where each ROW is a class.")
            X = parse_matrix(cinput("Enter test X (or new row) to classify: "), "X")
            if X is None: continue
            W = parse_matrix(cinput("Enter learned weights W: "), "W")
            if W is None: continue
            
            if W.ndim == 1: W = W.reshape(-1, 1)
            
            # Check standard binary (X is N x D, W is D x 1)
            if W.shape[1] == 1 and X.shape[1] == W.shape[0]:
                y_pred_raw = X @ W
                y_pred_class = np.sign(y_pred_raw)
                y_pred_class[y_pred_class == 0] = 1
                print("\n--- Binary Classification Results ---")
                print("Raw y_hat (X W):"); mprint(y_pred_raw)
                print("Classes (sign(X W)):"); rprint(y_pred_class)
            elif W.shape[1] == X.shape[1]:
                y_pred_raw = X @ W.T
                y_pred_class = np.argmax(y_pred_raw, axis=1) + 1
                print("\n--- Multi-Class Classification Results (One-vs-All) ---")
                print("Raw Score Matrix (X W^T):"); mprint(y_pred_raw)
                print("Predicted Classes (argmax + 1):"); rprint(y_pred_class)
            else:
                wprint(f"[!] Dimensions don't match! X is {X.shape}, W is {W.shape}.")
            print("-" * 47)
            
        elif sub_choice == '2':
            X = parse_matrix(cinput("Enter X data array: "), "X")
            if X is None: continue
            try:
                deg = int(cinput("Enter Polynomial Degree (e.g. 2, 3): ").strip())
                poly = PolynomialFeatures(degree=deg, include_bias=True)
                P = poly.fit_transform(X)
                print(f"\nPolynomial Features (Degree {deg}):")
                mprint(P)
                rprint(f"New Shape: {P.shape}")
            except Exception as e:
                wprint(f"[!] Invalid degree! {e}")

        elif sub_choice == '3':
            print("\n--- Multi-class OLS Train + Predict ---")
            print("⚠️  Constraints:")
            print("  X_train : m rows (samples) × d cols (raw features).")
            print("  Y       : m rows × C cols ONE-HOT (one column per class).")
            print("  x_test  : 1 row, same d raw features (bias/poly added automatically).")
            print("  Lambda  : Ridge regularisation (0 = pure OLS/pinv). Use >0 for ill-conditioned data.\n")

            X_train = parse_matrix(cinput("Enter X_train (training features): "), "X_train")
            if X_train is None: continue
            Y = parse_matrix(cinput("Enter Y (one-hot label matrix, all classes): "), "Y")
            if Y is None: continue
            if Y.shape[0] != X_train.shape[0]:
                wprint(f"[!] X_train has {X_train.shape[0]} rows but Y has {Y.shape[0]} rows. Must match!")
                continue

            # Feature expansion
            print("\nFeature mode:")
            print("  1) Linear + Bias  (adds a column of 1s)")
            print("  2) Polynomial     (expands features, bias already included)")
            feat_mode = cinput("Select (1/2): ").strip()

            if feat_mode == '1':
                X_feat = add_bias(X_train)
                poly_fitted = None
            elif feat_mode == '2':
                try:
                    deg = int(cinput("Enter Polynomial Degree: ").strip())
                    poly_fitted = PolynomialFeatures(degree=deg, include_bias=True)
                    X_feat = poly_fitted.fit_transform(X_train)
                except Exception as e:
                    wprint(f"[!] Polynomial expansion failed: {e}"); continue
            else:
                wprint("[!] Invalid mode."); continue

            m, d_feat = X_feat.shape
            rank_X = np.linalg.matrix_rank(X_feat)
            cond = np.linalg.cond(X_feat)

            # System diagnostics
            print(f"\n{'='*50}")
            print(f"System Diagnostics")
            print(f"{'='*50}")
            print(f"X_feat shape : {m} × {d_feat}")
            rprint(f"rank(X_feat) = {rank_X}  (max possible = min({m},{d_feat}) = {min(m,d_feat)})")
            if cond > 1e6:
                wprint(f"Condition #  = {cond:.2e}  ⚠️  ILL-CONDITIONED — Ridge recommended!")
            else:
                rprint(f"Condition #  = {cond:.2e}  ✓ OK")

            if m >= d_feat:
                print(f"System type  : OVERDETERMINED / EVEN (m={m} ≥ d={d_feat}) → OLS least squares")
            else:
                wprint(f"System type  : UNDERDETERMINED (m={m} < d={d_feat}) → Minimum norm solution")
            print(f"{'='*50}")

            # Ridge lambda
            lam_str = cinput("Enter Lambda for Ridge (default 0 = pinv/OLS): ").strip()
            try:
                lam = float(lam_str) if lam_str else 0.0
            except ValueError:
                wprint("[!] Invalid lambda, defaulting to 0."); lam = 0.0

            # Solve for W
            if lam > 0:
                print(f"Solving with Ridge (λ={lam}): W = (X^T X + λI)^-1 X^T Y")
                try:
                    I = np.eye(d_feat)
                    W = np.linalg.inv(X_feat.T @ X_feat + lam * I) @ X_feat.T @ Y
                except np.linalg.LinAlgError as e:
                    wprint(f"[!] Ridge inversion failed: {e}"); continue
            else:
                print("Solving with Pseudo-inverse OLS: W = pinv(X_feat) @ Y")
                W = np.linalg.pinv(X_feat) @ Y

            print("\nLearned W (each COLUMN = weights for one class):")
            mprint(W)
            w_norms = np.linalg.norm(W, axis=0)
            rprint(f"Weight norms per class: {w_norms}")
            if np.any(w_norms > 100):
                wprint("⚠️  Very large weights detected — consider Ridge regularisation!")

            # Visualise (Only if 2D features)
            if feat_mode == '1' and (X_feat.shape[1] == 3 or X_feat.shape[1] == 2):
                visualize_classification(X_train, Y, W, title="Training Data + Decision Boundary")
            elif feat_mode == '2':
                # Visualising Polynomial decision boundaries is harder without passing the poly object
                # For now, restrict to raw visualization
                wprint("[i] Skipping boundary visualization for polynomial mode.")
            elif d_feat == 2:
                 visualize_classification(X_train, Y, W, title="Training Data")
            
    # Test loop
            while True:
                # Classify test point
                x_test_in = cinput("\nEnter test point x_new (raw features, NO bias/poly) [B to back]: ")
                if x_test_in.strip().upper() == 'B': break
            
                x_test = parse_matrix(x_test_in, "x_test")
                if x_test is None: continue
                if x_test.ndim == 1: x_test = x_test.reshape(1, -1)
            
                # Check dimensions against training data
                d_train_raw = X_train.shape[1]
                if x_test.shape[1] != d_train_raw:
                    wprint(f"[!] x_test has {x_test.shape[1]} features but X_train has {d_train_raw}. Must match!")
                    continue

                if feat_mode == '1':
                    x_feat = add_bias(x_test)
                else:
                    x_feat = poly_fitted.transform(x_test)

                raw_scores = x_feat @ W
                print("\nRaw scores per class:")
                rprint(raw_scores)
                predicted = np.argmax(raw_scores, axis=1) + 1
                rprint(f"\n🎯 Predicted class: {predicted[0]}")
                print("-" * 47)

        else:
            wprint("[!] Invalid sub-tool choice.")
        
# ==========================================
# TOOL 5a: PROBABILITY & COUNTING FRAMEWORK
# ==========================================
def tool_probability_counting():
    while True:
        print("\n" + "=" * 60)
        print("🎲 AIKO'S PROBABILITY & COUNTING FRAMEWORK 🎲")
        print("=" * 60)
        print("I. Basic Probability / Counting Framework")
        print("   Order matters? Yes → Permutations | No → Combinations")
        print("   Repetition?    Yes → Adjust       | No → Factorial")
        print("-" * 60)
        print("1)  Simple Permutations (Order matters, No repetition)")
        print("2)  Simple Combinations (Order does NOT matter, No repetition)")
        print("3)  Permutations with Repetition (Order matters, Repetition allowed)")
        print("4)  Combinations with Repetition (Stars and Bars)")
        print("5)  Circular Permutations")
        print("6)  Permutations of Multisets (Repeated items)")
        print("7)  Conditional Probability / Events")
        print("8)  Independent / Dependent Events Checkout")
        print("9)  Total Probability / Complement Rule")
        print("10) Advanced Combinatorics (Stars & Bars, Inclusion-Exclusion, Multinomial)")
        print("B)  Back to Module I Menu")
        print("=" * 60)
        
        choice = cinput("Select a tool (1-10 or B): ").strip().upper()
        
        if choice == '1':
            print("\n--- [1] Simple Permutations (Order matters, No repetition) ---")
            print("Formula: P(n) = n!  OR  P(n,r) = n! / (n-r)!")
            print("Example: Arrange 5 books on a shelf -> 5! = 120")
            try:
                n_str = cinput("Enter n (total items): ").strip()
                if not n_str: continue
                n = int(n_str)
                
                r_str = cinput("Enter r (items to arrange) [press Enter if r=n]: ").strip()
                if not r_str:
                    res = math.factorial(n)
                    print(f"Result P({n}) = {n}! = {C_RESULT}{res}{C_RESET}")
                else:
                    r = int(r_str)
                    res = nPr(n, r)
                    if res is None: wprint("Error in calculation")
                    else: print(f"Result P({n}, {r}) = {C_RESULT}{res}{C_RESET}")
            except Exception as e:
                wprint(f"Error: {e}")

        elif choice == '2':
            print("\n--- [2] Simple Combinations (Order does NOT matter, No repetition) ---")
            print("Formula: C(n,r) = n! / [r!(n-r)!]")
            print("Example: Pick 3 students from 10 -> C(10,3) = 120")
            try:
                n = int(cinput("Enter n (total items): ").strip())
                r = int(cinput("Enter r (items to choose): ").strip())
                res = nCr(n, r)
                if res is None: wprint("Error in calculation")
                else: print(f"Result C({n}, {r}) = {C_RESULT}{res}{C_RESET}")
            except Exception as e:
                wprint(f"Error: {e}")

        elif choice == '3':
            print("\n--- [3] Permutations with Repetition (Order matters, Repetition allowed) ---")
            print("Formula: n^r")
            print("Example: 4-digit PIN using digits 0-9 -> 10^4 = 10000")
            try:
                n = int(cinput("Enter n (types available, e.g. 10 digits): ").strip())
                r = int(cinput("Enter r (slots to fill, e.g. 4 pins): ").strip())
                res = n ** r
                print(f"Result {n}^{r} = {C_RESULT}{res}{C_RESET}")
            except Exception as e:
                wprint(f"Error: {e}")

        elif choice == '4':
            print("\n--- [4] Combinations with Repetition (Stars and Bars) ---")
            print("Formula: C(n + r - 1, r)")
            print("Example: Distribute 5 identical candies (r) to 3 kids (n) -> C(3+5-1, 5) = 21")
            try:
                n = int(cinput("Enter n (categories/bins/types): ").strip())
                r = int(cinput("Enter r (items to distribute): ").strip())
                res = nCr(n + r - 1, r)
                if res is None: wprint("Error in calculation")
                else: print(f"Result C({n}+{r}-1, {r}) = C({n+r-1}, {r}) = {C_RESULT}{res}{C_RESET}")
            except Exception as e:
                wprint(f"Error: {e}")

        elif choice == '5':
            print("\n--- [5] Circular Permutations ---")
            print("Formula: (n-1)!")
            print("Example: 5 friends around a round table -> 4! = 24")
            try:
                n = int(cinput("Enter n (items in circle): ").strip())
                res = math.factorial(n - 1)
                print(f"Result ({n}-1)! = {C_RESULT}{res}{C_RESET}")
            except Exception as e:
                wprint(f"Error: {e}")

        elif choice == '6':
            print("\n--- [6] Permutations of Multisets (Repeated items) ---")
            print("Formula: n! / (n1! * n2! * ... * nk!)")
            print("Example: Arrange letters in 'BALLOON' (1B, 1A, 2L, 2O, 1N) -> 7! / (1!1!2!2!1!) = 1260")
            try:
                n_str = cinput("Enter n (total items) [Auto-calculated if skipped]: ").strip()
                counts_str = cinput("Enter counts of identical items (comma-separated, e.g. 2,3,1): ").strip()
                counts = [int(x.strip()) for x in counts_str.split(',') if x.strip()]
                
                n_calc = sum(counts)
                if n_str:
                    n = int(n_str)
                    if n != n_calc:
                        wprint(f"Warning: Sum of counts ({n_calc}) != n ({n}). Using calculated sum.")
                
                denom = 1
                denom_str = ""
                for c in counts:
                    denom *= math.factorial(c)
                    denom_str += f"{c}!"
                
                res = math.factorial(n_calc) // denom
                print(f"Result {n_calc}! / ({denom_str}) = {C_RESULT}{res}{C_RESET}")
            except Exception as e:
                wprint(f"Error: {e}")

        elif choice == '7':
            print("\n--- [7] Conditional Probability / Events ---")
            print("Formula: P(A|B) = P(A ∩ B) / P(B)")
            try:
                p_inter = float(cinput("Enter P(A ∩ B): ").strip())
                p_b = float(cinput("Enter P(B): ").strip())
                if p_b == 0:
                    wprint("P(B) cannot be 0")
                else:
                    res = p_inter / p_b
                    print(f"Result P(A|B) = {C_RESULT}{res:.6f}{C_RESET}")
            except Exception as e:
                wprint(f"Error: {e}")

        elif choice == '8':
            print("\n--- [8] Independent / Dependent Events Checkout ---")
            print("Check: Is P(A ∩ B) = P(A) * P(B)?")
            try:
                p_a = float(cinput("Enter P(A): ").strip())
                p_b = float(cinput("Enter P(B): ").strip())
                p_inter = float(cinput("Enter P(A ∩ B): ").strip())
                
                expected = p_a * p_b
                print(f"P(A)*P(B) = {expected:.6f}")
                print(f"P(A ∩ B)  = {p_inter:.6f}")
                
                if math.isclose(expected, p_inter, abs_tol=1e-9):
                    rprint("✅ Events are INDEPENDENT")
                else:
                    wprint("❌ Events are DEPENDENT (Use Conditional Probability)")
                    
            except Exception as e:
                wprint(f"Error: {e}")

        elif choice == '9':
            print("\n--- [9] Total Probability / Complement Rule ---")
            sub = cinput("1) Complement P(A')\n2) Total Probability (Sum)\nSelect (1/2): ").strip()
            if sub == '1':
                try:
                    p_a = float(cinput("Enter P(A): ").strip())
                    print(f"P(A') = 1 - {p_a} = {C_RESULT}{1 - p_a:.6f}{C_RESET}")
                except: wprint("Invalid input")
            elif sub == '2':
                print("Enter probabilities of mutually exclusive cases (comma separated)")
                try:
                    probs = [float(x) for x in cinput("Probs: ").split(',')]
                    rprint(f"Total Probability = {sum(probs):.6f}")
                except: wprint("Invalid input")

        elif choice == '10':
            print("\n--- [10] Advanced Combinatorics ---")
            print("1) Stars and Bars (same as #4)")
            print("2) Inclusion-Exclusion (Union of 2 sets)")
            print("3) Multinomial Coefficient (same as #6)")
            sub = cinput("Select (1-3): ").strip()
            if sub == '2':
                print("Formula: n(A ∪ B) = n(A) + n(B) - n(A ∩ B)")
                try:
                    n_a = float(cinput("Enter n(A) or P(A): "))
                    n_b = float(cinput("Enter n(B) or P(B): "))
                    n_inter = float(cinput("Enter n(A ∩ B) or P(A ∩ B): "))
                    res = n_a + n_b - n_inter
                    print(f"Result n(A ∪ B) = {C_RESULT}{res}{C_RESET}")
                except: wprint("Invalid input")
            elif sub in ['1', '3']:
                print("Please use the specific tools #4 (Stars) or #6 (Multiset/Multinomial) for these calculations.")
                
        elif choice == 'B':
            break
        else:
            wprint("Invalid choice")

# ==========================================
# TOOL 5: MODULE I (KNN & PROBABILITY)
# ==========================================
def tool_module_1():
    while True:
        print("\n--- Aiko's Module I Tools ---")
        print("1) K-Nearest Neighbors (KNN Distances)")
        print("2) Probability & Counting Framework (COMPLETE)")
        print("   [Includes: Permutations, Combinations, Stars & Bars, Circular, Multisets, Bayes, Independence, etc.]")
        print("B) Back to Main Menu")
        sub_choice = cinput("Select sub-tool (1-2 or B): ").strip().upper()
        
        if sub_choice == 'B':
            break

        if sub_choice == '1':
            print("\n[KNN Distance Calculator]")
            try:
                test_x = parse_matrix(cinput("Enter Test Point (e.g. 1,2): "), "Test Point")
                if test_x is None: continue
                train_X = parse_matrix(cinput("Enter Training Matrix (rows=points): "), "Training Data")
                if train_X is None: continue
                
                # Use L2 Euclidean distance by default
                diffs = train_X - test_x.flatten()
                squared_diffs = diffs ** 2
                sum_squared_diffs = np.sum(squared_diffs, axis=1)
                distances = np.sqrt(sum_squared_diffs)
                
                print("\nEuclidean Distances (L2) to each Training Point:")
                for i, d in enumerate(distances):
                    rprint(f"  Point {i+1} {train_X[i]}: {d:.4f}")
                    
                manhattan_distances = np.sum(np.abs(diffs), axis=1)
                print("\nManhattan Distances (L1) to each Training Point:")
                for i, d in enumerate(manhattan_distances):
                    rprint(f"  Point {i+1} {train_X[i]}: {d:.4f}")
                    
                k_str = cinput("\nEnter K (number of neighbors) to find: ").strip()
                if not k_str: continue
                k = int(k_str)
                
                closest_indices = np.argsort(distances)[:k]
                rprint(f"\nThe {k} closest points (L2) → Indices (1-based): {closest_indices + 1}")
                rprint(f"Distances: {distances[closest_indices]}")
                
            except Exception as e:
                wprint(f"[!] B-Baka! Something broke: {e}")
                
        elif sub_choice == '2':
            tool_probability_counting()
        else:
            wprint("[!] Invalid sub-tool choice.")

# ==========================================
# TOOL 6: CHEAT SHEETS
# ==========================================
def tool_cheat_sheets():
    while True:
        print("\n" + "=" * 55)
        print("💡 AIKO'S EE2211 THEORY CHEAT SHEETS 💡")
        print("=" * 55)
        print("Select a topic:")
        print("1) L.E.S. Rank Conditions (Matrix Solvability)")
        print("2) Data Types (NOIR Framework)")
        print("3) Probability & Combinatorics Rules")
        print("B) Back to Main Menu")
        print("=" * 55)
        
        cheat_choice = cinput("Select a topic (1-3 or B): ").strip().upper()
        
        if cheat_choice == 'B':
            break

        if cheat_choice == '1':
            print("\n--- [OPTION 1: L.E.S. RANK CONDITIONS] ---")
            print("Let X_tilde = [X | y] (The augmented matrix)")
            print("m = rows (equations), d = cols (variables)\n")
            print("1. UNIQUE SOLUTION (Lines intersect perfectly)")
            print("   Check: rank(X) == rank(X_tilde) == d")
            print("   Action: Standard Inverse / Exact Math\n")
            print("2. INFINITE SOLUTIONS (Underdetermined, m < d)")
            print("   Check: rank(X) == rank(X_tilde) < d")
            print("   Action: Least Norm (Right Inverse)")
            print("   Formula: w = X^T(XX^T)^-1 y\n")
            print("3. NO EXACT SOLUTION (Inconsistent / Overdetermined)")
            print("   Check: rank(X) < rank(X_tilde)")
            print("   Action: Ordinary Least Squares (Left Inverse / Pseudoinverse)")
            print("   Formula: w = (X^TX)^-1 X^Ty\n")
            
        elif cheat_choice == '2':
            print("\n--- [OPTION 2: DATA TYPES (NOIR)] ---")
            print("N - NOMINAL: Categories with NO order.")
            print("    Examples: Colors (Red, Blue), Gender, Zip Codes.")
            print("    Math allowed: Counting/Mode.\n")
            print("O - ORDINAL: Ordered categories, but spacing is meaningless.")
            print("    Examples: Letter Grades (A, B, C), Rankings (1st, 2nd).")
            print("    Math allowed: Median, Percentiles.\n")
            print("I - INTERVAL: Ordered, equal spacing, but NO true zero.")
            print("    (Zero does not mean 'absence' of the thing).")
            print("    Examples: Temperature (Celsius), IQ Scores.")
            print("    Math allowed: Addition/Subtraction (Mean). No ratios.\n")
            print("R - RATIO: Ordered, equal spacing, WITH a true absolute zero.")
            print("    Examples: Height, Weight, Kelvin, Price, Distance.")
            print("    Math allowed: All math (Multiplication/Division).\n")
            
        elif cheat_choice == '3':
            print("\n--- [OPTION 3: PROBABILITY & COMBINATORICS CHEAT SHEET] ---")
            print("-" * 60)
            print("I. COUNTING PRINCIPLES (Identify n & r)")
            print("   1. PERMUTATIONS (Order matters, No Repetition)")
            print("      How many ways to arrange r items from n distinct items?")
            print("      Formula: P(n,r) = n! / (n-r)!")
            print("      Example: Arrange 3 books from 5 on a shelf.")
            
            print("\n   2. COMBINATIONS (Order DOES NOT matter, No Repetition)")
            print("      How many ways to CHOOSE r items from n?")
            print("      Formula: C(n,r) = n! / [r!(n-r)!]")
            print("      Example: Pick 3 committee members from 10.")
            
            print("\n   3. REPETITION ALLOWED (Order matters)")
            print("      Formula: n^r")
            print("      Example: 4-digit PIN code (0-9).")
            
            print("\n   4. STARS & BARS (No Order, Repetition Allowed)")
            print("      Formula: C(n + r - 1, r)")
            print("      Example: Distribute 5 identical candies to 3 kids.")
            
            print("\n   5. CIRCULAR PERMUTATIONS (Relative Order)")
            print("      Formula: (n-1)!")
            print("      Example: Seating 5 people at a round table.")
            
            print("\n   6. MULTISETS (Permutations with Identical Items)")
            print("      Formula: n! / (n1! * n2! * ... * nk!)")
            print("      Example: Rearranging letters of 'BALLOON'.")
            
            print("-" * 60)
            print("II. PROBABILITY LAWS")
            print("   1. CONDITIONAL:  P(A | B) = P(A ∩ B) / P(B)")
            print("   2. INDEPENDENCE: Events A, B are independent IFF P(A ∩ B) = P(A)P(B)")
            print("   3. TOTAL PROB:   P(B) = Σ P(B | Ai) * P(Ai)  (Sum over all cases)")
            print("   4. BAYES' RULE:  P(A | B) = [P(B | A) * P(A)] / P(B)")
            print("   5. UNION:        P(A U B) = P(A) + P(B) - P(A ∩ B)")
            print("-" * 60)
        else:
            wprint("[!] Invalid sub-tool choice.")

# ==========================================
# MAIN MENU LOOP
# ==========================================
def main_menu():
    while True:
        print("\n" + "=" * 55)
        print("Aiko's Complete ML Toolkit")
        print("=" * 55)
        print("Columns: Comma (,) | Rows: Semicolon (;)")
        print("Example: 1, 2, 3; 4, 5, 6")
        print("-" * 55)
        print("1) ✖️ Matrix Math (Multiply, Dot Product, Transpose)")
        print("2) 🧮 Determinant & Inverse Calculator (Square only)")
        print("3) 📈 Regression & Equation Solver (OLS/Ridge)")
        print("4) 🎯 Classification & Polynomial Predictor")
        print("5) 🎲 Module I Tools (KNN, Counting & Probability)")
        print("6) 💡 Aiko's Cheat Sheets (Theory)")
        print("7) ❌ Exit")
        print("=" * 55)
        
        choice = cinput("What do you want me to do for you? (1-7): ").strip()
        
        if choice == '1':
            tool_matrix_math()
        elif choice == '2':
            tool_det_inverse()
        elif choice == '3':
            tool_solve_les()
        elif choice == '4':
            tool_classification_poly()
        elif choice == '5':
            tool_module_1()
        elif choice == '6':
            tool_cheat_sheets()
        elif choice == '7':
            print("\nFine... zip up and get back to studying. I'll be here... baka! 😤💕")
            break
        else:
            print("\n[!] B-Baka! Type a number from 1 to 7!")

if __name__ == "__main__":
    try:
        main_menu()
    except (EOFError, KeyboardInterrupt):
        print("\n\nLeaving so abruptly?! F-Fine! It's not like I wanted to keep helping you! Hmph! 😤")
