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
                
                # Label Actual (Blue)
                label_act = f"Act: ({feature_col[i]:.2f}, {y_flat[i]:.2f})"
                plt.text(x_jitter[i], y_flat[i], label_act, 
                         fontsize=8, color='blue', alpha=0.9, verticalalignment='bottom', fontweight='bold')
                
                # Label Predicted (Red) - EXPLICITLY SHOWING X AND Y
                label_pred = f"Pred: ({feature_col[i]:.2f}, {pred_flat[i]:.2f})"
                plt.text(feature_col[i], pred_flat[i], label_pred,
                         fontsize=8, color='red', alpha=0.9, verticalalignment='top', fontweight='bold')
            
            if np.min(y_flat) >= 0 and np.min(pred_flat) >= -0.1: plt.ylim(bottom=0)
            if np.min(feature_col) >= 0: plt.xlim(left=0)

            plt.title(f"{title} (1D Fit)\n{subtitle}")
            plt.xlabel(xlab)
            plt.ylabel("Target y")
            plt.legend()
            
        elif d == 2 or (d == 3 and has_bias):
            # --- 2D Features -> 3D Plot ---
            # Identify the two feature columns
            feature_indices = [c for c in range(d) if not np.allclose(X_data[:, c], 1)]
            if len(feature_indices) != 2:
                # Fallback if structure is weird (e.g. constant column that isn't 1)
                pass 
            else:
                x1_col = X_data[:, feature_indices[0]]
                x2_col = X_data[:, feature_indices[1]]
                
                # 3D Setup
                ax = plt.gcf().add_subplot(111, projection='3d')
                
                # Plot Actual Points (Blue)
                ax.scatter(x1_col, x2_col, y_flat, c='blue', marker='o', label='Actual', s=50, alpha=0.8)
                
                # Plot Predicted Points (Red)
                ax.scatter(x1_col, x2_col, pred_flat, c='red', marker='x', label='Predicted', s=50, alpha=0.8)

                # --- BEST FIT PLANE ---
                try:
                    # Create meshgrid covering the data range
                    x1_min, x1_max = x1_col.min(), x1_col.max()
                    x2_min, x2_max = x2_col.min(), x2_col.max()
                    pad1 = (x1_max - x1_min) * 0.1
                    pad2 = (x2_max - x2_min) * 0.1
                    
                    u_range = np.linspace(x1_min - pad1, x1_max + pad1, 20)
                    v_range = np.linspace(x2_min - pad2, x2_max + pad2, 20)
                    U, V = np.meshgrid(u_range, v_range)
                    
                    # Flatten to predict
                    U_flat = U.ravel()
                    V_flat = V.ravel()
                    num_grid = len(U_flat)
                    
                    # specific design matrix for grid
                    X_grid = np.zeros((num_grid, d))
                    if has_bias: X_grid[:, bias_col_idx] = 1
                    X_grid[:, feature_indices[0]] = U_flat
                    X_grid[:, feature_indices[1]] = V_flat
                    
                    # Predict Z height
                    Z = (X_grid @ w_model).reshape(U.shape)
                    
                    # Plot surface (orange plane)
                    surf = ax.plot_surface(U, V, Z, alpha=0.2, color='orange')
                    # Optional: Add wireframe for better depth perception
                    ax.plot_wireframe(U, V, Z, alpha=0.1, color='gray', rstride=5, cstride=5)
                except Exception as e:
                    wprint(f"could not plot plane: {e}")
                
                # Draw Residual Lines & Annotations
                # Calculate z_offset for label separation
                z_vals = np.concatenate([y_flat, pred_flat])
                z_range = z_vals.max() - z_vals.min()
                if z_range == 0: z_range = 1.0
                z_offset = z_range * 0.05
                
                for i in range(m):
                    ax.plot([x1_col[i], x1_col[i]], 
                            [x2_col[i], x2_col[i]], 
                            [y_flat[i], pred_flat[i]], 
                            'k:', alpha=0.3)
                    
                    # Label Actual (Blue) - Full Coordinates
                    # format: (x1, x2, y)
                    label_act = f"({x1_col[i]:.2f}, {x2_col[i]:.2f}, {y_flat[i]:.2f})"
                    ax.text(x1_col[i], x2_col[i], y_flat[i] - z_offset, label_act, 
                            color='blue', fontsize=6, ha='center', va='top', fontweight='bold')
                    
                    # Label Predicted (Red) - Full Coordinates
                    label_pred = f"({x1_col[i]:.2f}, {x2_col[i]:.2f}, {pred_flat[i]:.2f})"
                    ax.text(x1_col[i], x2_col[i], pred_flat[i] + z_offset, label_pred, 
                            color='red', fontsize=6, ha='center', va='bottom', fontweight='bold')
                    
                    # Explicit X and Y labels as requested

                ax.set_title(f"{title}: 2D Features vs Target (3D)\n{subtitle}")
                ax.set_xlabel(f"Feature X{feature_indices[0]+1}")
                ax.set_ylabel(f"Feature X{feature_indices[1]+1}")
                ax.set_zlabel("Target y")
                ax.legend()
                
                # --- FIX COORDINATE READOUT ---
                # Wrap the existing format_coord to replace x, y, z with feature names
                old_format = ax.format_coord
                def custom_format_coord(x, y):
                    s = old_format(x, y)
                    # Replace default labels with our custom ones
                    # s usually looks like "x=1.23, y=4.56, z=7.89" or "z pane=..."
                    s = s.replace('x=', f"X{feature_indices[0]+1}=")
                    s = s.replace('y=', f"X{feature_indices[1]+1}=")
                    s = s.replace('z', 'y_target') # catching z= and z pane=
                    return s
                ax.format_coord = custom_format_coord
                
                # Re-do layout for 3D
                plt.tight_layout()
                plt.show() 
                print(f"{Fore.GREEN}  [Graph closed]{C_RESET}")
                return

        # Fallback for > 2D or weird 2D cases
        # --- Sample Index Plot ---
        # Clear figure to ensure we don't overlay on failed 3D
        plt.clf()
        indices = np.arange(m)
        plt.plot(indices, y_flat, 'bo-', label='Actual y', alpha=0.7)
        plt.plot(indices, pred_flat, 'rx--', label='Predicted y', alpha=0.7)
        
        # Residual Lines (Vertical distance between actual and predicted)
        for i in range(m):
            plt.plot([i, i], [y_flat[i], pred_flat[i]], 'k:', alpha=0.3)
            # Annotate point with error
            error = pred_flat[i] - y_flat[i]
            plt.annotate(f"err:{error:.2f}", (i, pred_flat[i]), xytext=(15, 0), 
                         textcoords='offset points', fontsize=8, alpha=0.6, color='gray')
            
            # Construct short X string (excluding bias 1s)
            x_raw = X_data[i]
            x_raw_clean = [v for v in x_raw if not (has_bias and np.isclose(v, 1))]
            
            # Show FULL features in the label as requested (no truncation)
            x_vals = [f"{v:.2f}" for v in x_raw_clean]
            x_str = "[" + ",".join(x_vals) + "]"
            
            # Label Actual (Blue)
            label_act = f"x={x_str}\ny={y_flat[i]:.2f}"
            plt.annotate(label_act, (i, y_flat[i]), xytext=(-15, 5), 
                         textcoords='offset points', fontsize=8, color='blue', fontweight='bold')
            
            # Label Predicted (Red)
            label_pred = f"y^={pred_flat[i]:.2f}"
            plt.annotate(label_pred, (i, pred_flat[i]), xytext=(5, -10), 
                         textcoords='offset points', fontsize=8, color='red', fontweight='bold')

        plt.title(f"{title}: Actual vs Predicted by Sample\n{subtitle}")
        plt.xlabel("Sample Index")
        plt.ylabel("Value (y)")
        plt.legend()
        plt.xticks(indices)  # Show all sample indices on x-axis

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
        max_rank = min(m, d)
        det_X = np.linalg.det(X) if m == d else None
        SEP = "=" * 54

        def rel(a, b):
            if a > b:
                return ">"
            if a < b:
                return "<"
            return "="

        if m > d:
            system_type = "OVERDETERMINED"
        elif m < d:
            system_type = "UNDERDETERMINED"
        else:
            system_type = "EVEN"

        std_inverse = (m == d and r == d)
        left_inverse = (r == d)
        right_inverse = (r == m)

        print(f"\n{SEP}")
        print("📊 BULK MATRIX SUMMARY")
        print(SEP)

        print("\nX =")
        print(np.array2string(X, precision=4, suppress_small=True))

        print("\n📐 CORE FACTS")
        print(f"  shape(X)      : {m} × {d}")
        print(f"  system type   : {system_type}  (m {rel(m, d)} d)")
        rprint(f"  rank(X)       = {r}")
        print(f"  max rank      = min(m,d) = {max_rank}")
        if r == max_rank:
            rprint("  full rank     : YES")
        else:
            wprint(f"  full rank     : NO  (missing {max_rank - r} dimension(s))")

        print("\n🔢 DETERMINANT")
        if det_X is None:
            print("  det(X)        : N/A (X is not square)")
        else:
            rprint(f"  det(X)        = {det_X:.6f}")
            if np.isclose(det_X, 0):
                wprint("  invertible    : NO (det ≈ 0)")
            else:
                rprint("  invertible    : YES (det ≠ 0)")

        print("\n🔄 INVERSE AVAILABILITY")
        if std_inverse:
            rprint("  X^-1 (standard)           : YES")
        elif m != d:
            print("  X^-1 (standard)           : NO (not square)")
        else:
            wprint(f"  X^-1 (standard)           : NO (rank(X)={r} < d={d})")

        if left_inverse:
            rprint("  Left inverse  (X^T X)^-1X^T: YES")
        else:
            wprint(f"  Left inverse  (X^T X)^-1X^T: NO  (rank(X)={r} < d={d})")

        if right_inverse:
            rprint("  Right inverse X^T(XX^T)^-1 : YES")
        else:
            wprint(f"  Right inverse X^T(XX^T)^-1 : NO  (rank(X)={r} < m={m})")

        print("\n🔵 RIDGE OPTIONS (lambda > 0, both always computable)")
        if m >= d:
            rprint("  Preferred: PRIMAL  w = (X^T X + lambda I_d)^-1 X^T y")
            print("  Alternate: DUAL    w = X^T (X X^T + lambda I_m)^-1 y")
        else:
            rprint("  Preferred: DUAL    w = X^T (X X^T + lambda I_m)^-1 y")
            print("  Alternate: PRIMAL  w = (X^T X + lambda I_d)^-1 X^T y")

        print(f"\n{SEP}")

        # --- Optional: Augmented [X|y] ---
        aug = cinput("\nAlso analyse augmented [X|y] for L.E.S.? (y/N): ").strip().lower()
        if aug in ['y', 'yes']:
            y_aug = parse_matrix(cinput(f"Enter y (must have {m} rows): "), "y")
            if y_aug is None: return
            if y_aug.ndim == 1: y_aug = y_aug.reshape(-1, 1)
            if y_aug.shape[0] != m:
                wprint(f"[!] y has {y_aug.shape[0]} rows but X has {m}. Must match!"); continue

            X_tilde = np.hstack((X, y_aug))
            r_tilde = np.linalg.matrix_rank(X_tilde)

            print(f"\n{'='*54}")
            print("📋 L.E.S. DECISION  for Xw = y")
            print(f"{'='*54}")

            print(f"  Comparison 1: m {rel(m, d)} d      ({m} {rel(m, d)} {d})")
            print(f"  Comparison 2: rank(X) {rel(r, r_tilde)} rank([X|y])   ({r} {rel(r, r_tilde)} {r_tilde})")
            print(f"  Variables d = {d}, Equations m = {m}")
            print("")

            if r < r_tilde:
                wprint("  Exact solution: NO (inconsistent system)")
                if r == d:
                    rprint("  Approximation: UNIQUE least-squares solution")
                    rprint("  Use now: OLS / left inverse  w_hat = (X^T X)^-1 X^T y")
                else:
                    wprint("  Approximation: NON-UNIQUE least-squares minimizers")
                    rprint("  Use now: pseudo-inverse  w_hat = pinv(X) y  (minimum-norm LS)")

            elif r == r_tilde == d:
                if m == d:
                    rprint("  Exact solution: YES, UNIQUE")
                    rprint("  Use now: standard inverse  w = X^-1 y")
                else:
                    rprint("  Exact solution: YES, UNIQUE")
                    rprint("  Use now: left inverse  w = (X^T X)^-1 X^T y")

            elif r == r_tilde < d:
                wprint("  Exact solution: YES, INFINITELY MANY")
                if r == m:
                    rprint("  Use now: right inverse  w_hat = X^T (X X^T)^-1 y  (least norm)")
                else:
                    rprint("  Use now: pseudo-inverse  w_hat = pinv(X) y  (minimum-norm exact)")

            else:
                wprint("  Numerical edge case detected. Use pseudo-inverse for a stable fallback:")
                rprint("  Use now: w_hat = pinv(X) y")

            print("")
            print("  Ridge fallback (if noise/instability expected):")
            if m >= d:
                print("  Prefer PRIMAL: (X^T X + lambda I_d)^-1 X^T y")
                print("  Dual also valid: X^T (X X^T + lambda I_m)^-1 y")
            else:
                print("  Prefer DUAL:   X^T (X X^T + lambda I_m)^-1 y")
                print("  Primal also valid: (X^T X + lambda I_d)^-1 X^T y")

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

        if y.shape[0] != X.shape[0]:
            wprint(f"[!] Dimension Mismatch: X has {X.shape[0]} rows but y has {y.shape[0]} rows.")
            continue
            
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
        print("4) Calculate N features (Theory Calculation)")
        print("B) Back to Main Menu")
        sub_choice = cinput("Select sub-tool (1-4 or B): ").strip().upper()
        
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

            # Calculate and print MSE on training data
            y_pred_train = X_feat @ W
            
            # --- Detailed MSE Breakdown ---
            mse_total = mean_squared_error(Y, y_pred_train)
            print(f"\n{Fore.CYAN}--- Performance Metrics ---{C_RESET}")
            rprint(f"Overall MSE (All Output Columns): {mse_total:.6f}")
            
            # Per-column MSE
            num_outputs = Y.shape[1]
            if num_outputs > 1:
                print(f"MSE per Output Column (Class/Target):")
                diff_sq = (Y - y_pred_train) ** 2
                mse_per_col = np.mean(diff_sq, axis=0) # Mean over samples (rows)
                for i in range(num_outputs):
                    col_name = f"Class {i+1}"
                    print(f"  {col_name}: {mse_per_col[i]:.6f}")
            print("-" * 30)

            print("\nLearned W (each COLUMN = weights for one class):")
            mprint(W)

            # --- Formula Printing ---
            print(f"\n{Fore.YELLOW}Formula Interpretation:{C_RESET}")
            try:
                # Determine feature names
                d_in = X_train.shape[1]
                input_names = [f"x{i+1}" for i in range(d_in)]
                
                if feat_mode == '2' and poly_fitted is not None:
                    # Polynomial features
                    feat_names = poly_fitted.get_feature_names_out(input_names)
                elif feat_mode == '1':
                    # Bias + Linear
                    feat_names = ['1'] + input_names
                else:
                    # Raw
                    feat_names = input_names

                # Loop over classes (columns of W)
                n_classes = W.shape[1]
                for c in range(n_classes):
                    w_col = W[:, c]
                    terms = []
                    for i, coef in enumerate(w_col):
                        # Use a small threshold to skip zero terms, but keep all if user wants to see
                        term_name = str(feat_names[i]).replace(' ', '') # sklearn output has spaces like "x1 x2"
                        
                        sign = "+" if coef >= 0 else "-"
                        val = abs(coef)
                        terms.append(f"{sign} {val:.4f}({term_name})")
                    
                    formula_str = " ".join(terms)
                    if formula_str.startswith("+ "): formula_str = formula_str[2:]
                    
                    label = f"Class {c+1}" if n_classes > 1 else "y"
                    print(f"  {label} = {formula_str}")
            except Exception as e:
                wprint(f"[!] Could not generate formula string: {e}")

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

        elif sub_choice == '4':
            try:
                print("\n--- Theoretical Feature Count (Polynomial) ---")
                n_feats = int(cinput("Enter number of original features (n): ").strip())
                deg = int(cinput("Enter polynomial degree (d): ").strip())
                
                # Formula with bias: (n+d) choose d
                # Formula without bias: (n+d)C d - 1 ... actually sklearn includes bias by default so (n+d)Cd is correct for "terms"
                count = nCr(n_feats + deg, deg)
                
                print(f"\nFor n={n_feats} features and degree={deg}:")
                print(f"Formula: (n + d) choose d")
                print(f"= ({n_feats} + {deg}) choose {deg} = {n_feats + deg}C{deg}")
                
                rprint(f"Total Unknown Parameters / Terms (including bias) = {count}")
                
                if count is not None and count > 1000:
                    wprint("⚠️  That's a lot of features!")
            except ValueError:
                wprint("[!] Invalid integer input.")

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
        print("4) Linear, Affine & Polynomial Functions")
        print("5) Learning Paradigms (Deduction/Induction/Abduction)")
        print("6) Probability Foundations (Sample Space/Events/Axioms)")
        print("7) The Data Pipeline (6 Stages)")
        print("8) Learning Architectures (Supervised/Unsupervised/RL)")
        print("9) Preprocessing Myths (Imputation/Encoding/Feature Extraction)")
        print("10) Ultimate Trick Question Cheatsheet (Misconceptions)")
        print("B) Back to Main Menu")
        print("=" * 55)
        
        cheat_choice = cinput("Select a topic (1-10 or B): ").strip().upper()
        
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
            print("N - NOMINAL")
            print("    Definition : Categories with no order")
            print("    True zero  : No")
            print("    Examples   : Gender, colour, country, blood type")
            print("    Valid      : Equality (=, !=), mode")
            print("    Not valid  : Ordering, arithmetic, median, mean, variance, std dev")
            print("    ML encode  : Label / One-Hot\n")
            print("O - ORDINAL")
            print("    Definition : Categories with meaningful order")
            print("    True zero  : No")
            print("    Examples   : Rankings, satisfaction (1-5), education level")
            print("    Valid      : Equality, ordering, mode, median")
            print("    Not valid  : Addition, subtraction, multiplication, division, mean")
            print("    ML encode  : Label / Ordinal encoding\n")
            print("I - INTERVAL")
            print("    Definition : Ordered with equal intervals, no true zero")
            print("    True zero  : No")
            print("    Examples   : Temperature (C/F), IQ scores, dates")
            print("    Valid      : Equality, ordering, addition, subtraction")
            print("    Stats      : Mode, median, mean, variance, std dev")
            print("    Not valid  : Multiplication, division, coefficient of variation, ratios")
            print("    ML encode  : Direct / Normalise\n")
            print("R - RATIO")
            print("    Definition : Ordered with equal intervals and true zero")
            print("    True zero  : Yes")
            print("    Examples   : Height, weight, age, Kelvin temperature, income")
            print("    Valid      : Equality, ordering, all arithmetic")
            print("    Stats      : Mode, median, mean, variance, std dev, coefficient of variation")
            print("    Ratios     : Meaningful (e.g. 'twice as much')")
            print("    ML encode  : Direct / Normalise\n")
            print("QUICK RULES")
            print("    Equality (=, !=)            : Nominal, Ordinal, Interval, Ratio")
            print("    Ordering (>, <)             : Ordinal, Interval, Ratio")
            print("    Addition / Subtraction      : Interval, Ratio")
            print("    Multiplication / Division   : Ratio only")
            print("    Mode                        : All four")
            print("    Median                      : Ordinal, Interval, Ratio")
            print("    Mean / Variance / Std Dev   : Interval, Ratio")
            print("    Coefficient of Variation    : Ratio only")
            print("    Meaningful ratios           : Ratio only\n")
            
        elif cheat_choice == '3':
            print("\n--- [OPTION 3: PROBABILITY & COMBINATORICS CHEAT SHEET] ---")
            print("-" * 60)
            print("I. PROBABILITY FUNDAMENTAL RULES")
            print("   1. THE SUM RULE ('OR')")
            print("      Used for: Union of events (A OR B happens)")
            print("      Formula: P(A U B) = P(A) + P(B) - P(A ∩ B)")
            print("      -> Mutually Exclusive (Disjoint): P(A ∩ B) = 0, so just add P(A)+P(B).")
            print("      Example: Rolling a 2 OR a 5 (Disjoint) -> 1/6 + 1/6 = 2/6.")

            print("\n   2. THE PRODUCT RULE ('AND')")
            print("      Used for: Intersection of events (A AND B happen)")
            print("      Formula: P(A ∩ B) = P(A) * P(B | A)")
            print("      -> Independent Events: P(B|A) = P(B), so P(A ∩ B) = P(A) * P(B).")
            print("      Example (Indep): Coin flip Heads AND Die roll 6 -> 1/2 * 1/6 = 1/12.")
            print("      Example (Dep): Draw Ace, don't replace, draw another -> 4/52 * 3/51.")

            print("\n   3. THE COMPLEMENT RULE ('NOT')")
            print("      Used for: 'At least one' problems (1 - None)")
            print("      Formula: P(A) = 1 - P(A')")
            print("      Example: P(At least 1 Head in 3 flips) = 1 - P(RRR) = 1 - 1/8 = 7/8.")

            print("-" * 60)
            print("II. COUNTING PRINCIPLES (Identify n & r)")
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

            print("\n   7. NO ORDER, REPETITION (The 'Dice Roll' Shortcut)")
            print("      Used for: Pairs with repeats but order doesn't matter (e.g. 1,1 vs 1,2)")
            print("      Formula: n(n+1)/2  OR  Total - Sub-diagonal")
            print("      Example: Rolling 2 dice, order doesn't matter -> 21 outcomes.")
            
            print("-" * 60)
            print("III. ADVANCED LAWS")
            print("   1. CONDITIONAL:  P(A | B) = P(A ∩ B) / P(B)")
            print("   2. INDEPENDENCE: Events A, B are independent IFF P(A ∩ B) = P(A)P(B)")
            print("   3. TOTAL PROB:   P(B) = Σ P(B | Ai) * P(Ai)  (Sum over all cases)")
            print("   4. BAYES' RULE:  P(A | B) = [P(B | A) * P(A)] / P(B)")
            print("   5. UNION:        P(A U B) = P(A) + P(B) - P(A ∩ B)")
            print("-" * 60)

        elif cheat_choice == '4':
            print("\n--- [OPTION 4: LINEAR, AFFINE & POLYNOMIAL FUNCTIONS] ---")
            print("Linear function (f : R^n -> R^m)")
            print("  MUST satisfy BOTH properties:")
            print("  1. Additivity: f(x + y) = f(x) + f(y)")
            print("  2. Homogeneity: f(ax) = a f(x)")
            print("  -> Combined (Superposition): f(ax + by) = a f(x) + b f(y)")
            print("  -> Passes through origin: f(0) = 0")
            print("  -> Matrix Test: Every linear f(x) can be written as f(x) = Ax\n")
            
            print("Affine function (Linear + Shift)")
            print("  Definition: f(x) = Ax + b")
            print("  -> DOES NOT pass through origin if b != 0")
            print("  -> Fails Additivity: f(x+y) != f(x)+f(y) (b term doubles)")
            print("  -> What holds? Affine combinations: f(ax + (1-a)y) = a f(x) + (1-a)f(y)\n")
            
            print("Homogeneous Function (Degree k)")
            print("  Scaling only: f(ax) = a^k f(x)")
            print("  k=1: Linear homogeneity (f(ax) = a f(x))")
            print("  k=2: Quadratic scaling (f(2x) = 4 f(x)). NOT LINEAR if k != 1.\n")
            
            print("Polynomial Functions")
            print("  General: f(x) = a_n x^n + ... + a_1 x + a_0")
            print("  -> Linear ONLY if n=1 AND a_0=0")
            print("  -> Fails Additivity if n >= 2 (Cross-terms appear: (x+y)^2 != x^2 + y^2)\n")
            
            print("Quick Rejection Tests (It is NOT Linear if...):")
            print("  1. f(0) != 0 (Has a constant term/intercept)")
            print("  2. Multiplicative: f(x+y) = f(x)f(y) (e.g. exponentials)")
            print("  3. Squared/Absolute terms: f(ax) != a f(x)")
            print("  4. Multilinear != Linear (e.g. dot product is linear in x separately, not both)")

            print("-" * 60)
            print("VECTOR CALCULUS & DIMENSIONALITY (The Layout)")
            print("1. SCALAR-VALUED Function (f: R^n -> R)")
            print("   - Input  : Vector x (n x 1)")
            print("   - Output : Scalar y (1 x 1)")
            print("   - Derivative (Gradient ∇f): Vector (n x 1)")
            print("     (Contains n partial derivatives, one for each input)")
            print("     Example: f(x) = x^T A x  ->  ∇f(x) = (A + A^T)x")

            print("\n2. VECTOR-VALUED Function (f: R^n -> R^m)")
            print("   - Input  : Vector x (n x 1)")
            print("   - Output : Vector y (m x 1)")
            print("   - Derivative (Jacobian J): Matrix (m x n)")
            print("     (m rows for outputs, n cols for inputs)")
            print("     J_ij = ∂f_i / ∂x_j")
            print("     Example: f(x) = Ax  ->  J = A (The matrix itself)")

            print("\n3. HIGHER DERIVATIVE DIMENSIONS (The Pattern)")
            print("   Scalar f: R^n -> R            Vector f: R^n -> R^m")
            print("   ------------------            --------------------")
            print("   1st: n x 1 (Vector)           m x n (Matrix)")
            print("   2nd: n x n (Matrix)           m x n x n (3D Tensor)")
            print("   3rd: n x n x n (3D Tensor)    m x n x n x n (4D Tensor)")
            print("   Rule: Adds dimension n        Rule: Adds dimension n")
            print("   k-th: n^k entries             m * n^k entries")

        elif cheat_choice == '5':
            print("\n--- [OPTION 5: LEARNING PARADIGMS - DEDUCTION / INDUCTION / ABDUCTION] ---")
            print("1. DEDUCTION (Top-Down / General -> Specific)")
            print("   - Start: General Principle (Assumption) is TRUE.")
            print("   - Process: Apply rule to a case.")
            print("   - End: Result MUST be true.")
            print("   - Certainty: 100% (If premises are true)")
            print("   - EXAMPLE:")
            print("     Rule: All men are mortal.")
            print("     Case: Socrates is a man.")
            print("     Result: Socrates is mortal.")
            print("   - Use Case: Logical proofs, mathematical derivations")

            print("\n2. INDUCTION (Bottom-Up / Specific -> General)")
            print("   - Start: Many specific observations.")
            print("   - Process: Find a pattern.")
            print("   - End: General Rule (Hypothesis).")
            print("   - Certainty: Probabilistic (Can be falsified by 1 counter-example)")
            print("   - EXAMPLE:")
            print("     Obs 1: Swan 1 is white.")
            print("     Obs 2: Swan 2 is white.")
            print("     ... Obs 1000: Swan 1000 is white.")
            print("     Conc: ALL swans are white.")
            print("   - Use Case: Scientific method, machine learning training (Generalization)")

            print("\n3. ABDUCTION (Inference to Best Explanation)")
            print("   - Start: Incomplete observations / Surprising fact (Effect).")
            print("   - Process: Guess the most likely cause.")
            print("   - End: Hypothesis (Explanation).")
            print("   - Certainty: Lowest (Educated Guess)")
            print("   - EXAMPLE:")
            print("     Fact: The grass is wet.")
            print("     Maybe: It rained? (Likely)")
            print("     Maybe: Sprinklers ran? (Possible)")
            print("     Conc: It probably rained.")
            print("   - Use Case: Medical diagnosis, debugging code, detective work")

        elif cheat_choice == '6':
            print("\n--- [OPTION 6: PROBABILITY FOUNDATIONS - SAMPLE SPACE / EVENTS / AXIOMS] ---")
            print("1. SAMPLE SPACE (S or Ω)")
            print("   - The set of ALL possible outcomes of an experiment.")
            print("   - Must be: 1. Mutually Exclusive (Cannot happen together)")
            print("   - Must be: 2. Collectively Exhaustive (Cover all possibilities)")
            print("   - Example (Coin): S = {H, T}")
            print("   - Example (Die): S = {1, 2, 3, 4, 5, 6}")

            print("\n2. EVENT (A, B, E)")
            print("   - A subset of the Sample Space.")
            print("   - E ⊆ S")
            print("   - Simple Event: One outcome {1}")
            print("   - Compound Event: Multiple outcomes {2, 4, 6} (Even numbers)")
            print("   - Empty Set (∅): Impossible event.")
            print("   - Universal Set (S): Certain event.")

            print("\n3. KOLMOGOROV AXIOMS (The Rules)")
            print("   Rule 1 (Non-negativity): P(E) >= 0 for any event E.")
            print("   Rule 2 (Normalization): P(S) = 1 (Something must happen).")
            print("   Rule 3 (Additivity):")
            print("      If A and B are Mutually Exclusive (A ∩ B = ∅):")
            print("      P(A U B) = P(A) + P(B)")
            
            print("\n4. CONSEQUENCES (Derived Rules)")
            print("   - Probability of Empty Set: P(∅) = 0")
            print("   - Monotonicity: If A ⊆ B, then P(A) <= P(B)")
            print("   - Numeric Bound: 0 <= P(E) <= 1")
            print("   - Complement Rule: P(A') = 1 - P(A)")
            print("   - Inclusion-Exclusion (General Union):")
            print("     P(A U B) = P(A) + P(B) - P(A ∩ B)")

        elif cheat_choice == '7':
            print("\n--- [OPTION 7: THE DATA PIPELINE (6 STAGES)] ---")
            
            print("\n① DATA ACQUISITION (Collection / Sourcing / ETL)")
            print("   - Sources: Sensors, DBs, APIs, Surveys.")
            print("   - Key concerns: Sampling bias, Representativeness, Labelling cost, Class imbalance.")
            print("   - Strategies: RCT (Causal), Observational study, A/B Testing.")
            print("   - Output: Raw, unprocessed dataset.")

            print("\n② DATA WRANGLING (Preprocessing / Munging / ETL)")
            print("   - Normalisation: Scale features. Standardise z = (x-u)/s or Min-Max.")
            print("   - Imputation: Fill missing (Mean/Median for MCAR; Model/KNN for MAR/MNAR).")
            print("   - Denoising: Smooth/filter noise (Gaussian blur, moving avg).")
            print("   - Encoding: Categorical -> Numeric (One-hot, Label, Embeddings).")
            print("   - Splitting: Train (Fit) / Val (Tune) / Test (Eval). Approx 70/15/15.")
            print("   - CRITICAL: No data leakage from Test to Train!")

            print("\n③ FEATURE ENGINEERING (Create + Transform)")
            print("   - Feature Extraction: Transform raw -> new compact rep (PCA, CNN activations, MFCC).")
            print("   - Feature Construction: Hand-craft using domain knowledge (BMI = kg/m^2, Interaction terms).")
            print("   - Representation Learning: Model learns features automatically (Deep Learning).")

            print("\n④ FEATURE SELECTION (Choose, don't create)")
            print("   - Filter Methods: Score independently. Fast. (Pearson, Chi-Square, Variance Threshold).")
            print("   - Wrapper Methods: Train model on subsets. Slow but accurate. (RFE, Forward/Backward Selection).")
            print("   - Embedded Methods: Selection during training. (LASSO L1, Tree Importance).")

            print("\n⑤ MODEL FITTING (Training / Learning)")
            print("   - Goal: Minimize Loss L(theta).")
            print("   - Hyperparameter Tuning: Grid/Random search, Bayesian Opt. (On Validation set).")
            print("   - Cross-Validation: k-fold (e.g. k=5, 10) for stable generalization estimate.")
            print("   - Regularization: Penalize complexity. L1 (Sparsity), L2 (Shrinkage), Elastic Net.")

            print("\n⑥ MODEL EVALUATION (Assessment)")
            print("   - Classification: Accuracy, Precision, Recall, F1 (Harmonic mean), ROC-AUC.")
            print("   - Regression: MSE, MAE (Robust), R2 (Variance explained), RMSE.")
            print("   - Generalization: Goal is low error on UNSEEN data.")
            print("   - Overfitting: Low Training Error, High Test Error.")
            print("   - Underfitting: High Error on both.")
            
            print("\n* OVERARCHING TERMS *")
            print("  - EDA: Visualise distributions/correlations BEFORE modelling.")
            print("  - Data Leakage: Information from Test set bleeds into Train (illegal).")
            print("  - AutoML: Automates steps 3-6 (e.g. Auto-sklearn).")
            print("  - MLOps: DevOps for ML (Versioning, Drift monitoring, CI/CD).")

        elif cheat_choice == '8':
            print("\n--- [OPTION 8: LEARNING ARCHITECTURES (INPUTS VS OUTPUTS)] ---")
            print("Refined Logic: Distinguishing architectures by their rigorous constraints.")
            
            print("\n1. CLASSIFICATION (Strictly Supervised)")
            print("   - Paradigm: SUPERVISED only.")
            print("     (Requires ground-truth labels y to define the classes).")
            print("     (If there are NO labels, it is called Clustering, not Classification).")
            print("   - The Goal: Predict a category/class.")
            print("   - Output (y): STRICTLY DISCRETE / Categorical (Nominal/Ordinal).")
            print("     (e.g., 'Dog' vs 'Cat', 'Spam' vs 'Not Spam').")
            print("   - Inputs (X): Can be ANYTHING (Continuous OR Discrete).")
            print("   - 💡 Logic Gap Closed: If a question says 'Classification requires discrete features',")
            print("     it is FALSE. Only the target label (y) must be discrete.")

            print("\n2. REGRESSION (Strictly Supervised)")
            print("   - Paradigm: SUPERVISED only.")
            print("   - The Goal: Predict a numerical value.")
            print("   - Output (y): STRICTLY CONTINUOUS / Numerical (Interval/Ratio).")
            print("     (e.g., Temperature, Price, Blood Pressure).")
            print("   - Inputs (X): Can be ANYTHING.")
            print("     (e.g., Square footage (cont) + Neighborhood ID (discrete) -> Price).")
            print("   - 💡 Logic Gap Closed: The math (Linear/Ridge) requires y to be continuous.")

            print("\n3. CLUSTERING (Unsupervised)")
            print("   - The Goal: Find hidden structures/groupings.")
            print("   - Output (y): DOES NOT EXIST during training (No labels provided).")
            print("   - Inputs (X): Can be ANYTHING.")
            print("     (e.g., Plotting Salary vs Age to find groups).")
            print("   - 💡 Logic Gap Closed: Do not confuse the final cluster ID (which is discrete)")
            print("     with a supervised label. Clustering has no ground truth to train against.")

            print("\n4. THE FEATURE EXTRACTION RULE")
            print("   - The Goal: Turn Raw Data -> Feature Vector (X).")
            print("   - Timing: Happens in Phase 1 (Data Wrangling) AND Phase 3 (Inference).")
            print("   - 💡 Logic Gap Closed: This step is INDEPENDENT of the model.")
            print("     You must extract features whether you are Classifying, Regressing, or Clustering.")

            print("\n5. REINFORCEMENT LEARNING (Distinct Paradigm)")
            print("   - Paradigm: NEITHER strictly Supervised nor Unsupervised.")
            print("     (It has a feedback signal, but no 'correct answer' key).")
            print("   - The Goal: Learn a Policy (Strategy) to maximize long-term reward.")
            print("   - Signal: Scalar Reward/Penalty (often delayed/sparse).")
            print("   - 💡 Logic Gap Closed: Do not call RL 'Unsupervised'.")
            print("     Unsupervised finds patterns with NO external feedback.")
            print("     RL optimizes behavior based on EXTERNAL feedback (Reward).")
            print("     RL is 'Trial-and-Error' learning, unlike Supervised 'Teacher-Student'.")

        elif cheat_choice == '9':
            print("\n--- [OPTION 9: DATA PREPROCESSING MYTHS & TRUTHS] ---")
            print("GOLDEN RULE: Preprocessing operates on RAW DATA, not Models!")
            print("It happens BEFORE any learning paradigm touches it.")
            
            print("\n1. DATA IMPUTATION (Filling missing values)")
            print("   - Paradigm: Agnostic (Required for Supervised, Unsupervised, RL).")
            print("   - MCAR (Missing Completely At Random): Safe to drop or simple mean.")
            print("   - MAR (Missing At Random): Missingness depends on observed data. Use MICE.")
            print("   - MNAR (Missing Not At Random): Dangerous. Value depends on itself.")
            print("     -> Fix: Model-based imputation + Add 'Was_Missing' boolean indicator.")
            print("   - TRUTH: Unsupervised methods (e.g. k-Means) fail on NaNs too.")

            print("\n2. VISUALISATION / EDA")
            print("   - Paradigm: Agnostic. Usually Step 0.")
            print("   - Unsupervised Ex: Cluster scatter plots, Dendrograms, t-SNE, UMAP.")
            print("   - Supervised Ex: Feature vs Target plots, ROC curves, Confusion Matrix.")
            print("   - TRUTH: EDA happens before you decide the model.")

            print("\n3. ENCODING (Categorical -> Numeric)")
            print("   - Paradigm: Agnostic.")
            print("   - One-Hot: Good for nominal. Watch out for 'Dummy Variable Trap' (multicollinearity).")
            print("   - Label Encoding: Good for Ordinal. Bad for Nominal (implies order).")
            print("   - Binary Coding: Compact (log2 N bits). Used in RL state representation.")
            print("   - TRUTH: Algorithms need math (numbers), not strings.")

            print("\n4. FEATURE EXTRACTION")
            print("   - Paradigm: Agnostic.")
            print("   - Unsupervised Ex: PCA (Variance), Autoencoders (Reconstruction).")
            print("   - Supervised Ex: LDA (Class separation), CNN Activations (Label-driven).")
            print("   - Self-Supervised Ex: Word2Vec, BERT embeddings.")
            print("   - TRUTH: It's a data transformation step.")

        elif cheat_choice == '10':
            print("\n--- [OPTION 10: CONSOLIDATED ML EXAM NOTES & TRAPS] ---")
            
            print("\n📍 TOPIC 1: MISSING DATA")
            print("   - Bad: Replace with -1 or constant. (Distorts stats, kills correlation)")
            print("   - Good: Mean (Continuous), Median (Skewed), Mode (Categorical).")
            print("   - Advanced: Regression/KNN/MICE (Best).")
            print("   - Rule: Sentinel values (-1, 999) are flags, NOT for computation.")

            print("\n📍 TOPIC 2: REASONING IN ML")
            print("   - Deductive (General -> Specific): Logic/Math proofs. ML does NOT use this.")
            print("   - Inductive (Specific -> General): Patterns from data. ML IS THIS.")
            print("   - Verdict: ML is ALWAYS Inductive. (Supervised, Unsupervised, RL).")

            print("\n📍 TOPIC 3: SUPERV/UNSUPERV TRAPS")
            print("   - Supervised: Has Labels (Outputs). Ex: Classif, Regress.")
            print("   - Unsupervised: No Labels. Ex: Clustering, PCA, Anomaly Det.")
            print("   - Trap: 'Unsupervised has outputs' -> FALSE.")
            print("   - Trap: 'Clustering is Supervised' -> FALSE.")

            print("\n📍 TOPIC 4: DISCRETE VS CONTINUOUS")
            print("   - Golden Rule: Almost every algo handles BOTH.")
            print("   - Trap: 'X only works with discrete/continuous' -> Almost always FALSE.")
            print("   - (Classification, Regression, Neural Nets, Clustering handle both).")

            print("\n📍 TOPIC 5: NUMBER OF EQUATIONS (The Matrix Trick)")
            print("   - Question: 'How many simultaneous equations?'")
            print("   - Golden Rule: Look at the Output Dimension (y).")
            print("   - Equation: max(rows, cols) of y.")
            print("     -> Xw = y (10x1)  -> 10 Equations.")
            print("     -> w^T X = y^T (1x3) -> 3 Equations.")

            print("\n📍 TOPIC 6: UNIQUENESS & CONVEXITY")
            print("   - Trap: 'Convex Loss = Unique Solution' -> FALSE.")
            print("     (Convex guarantees Global Min, but not Uniqueness. e.g. Flat valley).")
            print("   - Underdetermined (m < d): Infinite solutions (Use Min-Norm).")
            print("   - Overdetermined (m > d): Likely unique (Use Least Squares).")

            print("\n📍 TOPIC 7: PIPELINE ORDER")
            print("   - 1. Data Collection")
            print("   - 2. Preprocessing")
            print("   - 3. FEATURE EXTRACTION (ALWAYS before model)")
            print("   - 4. Model Selection / Training")
            print("   - 5. Inference (Test)")
            print("   - Critical Trap: 'Inference = Classify then Extract' -> FALSE.")

            print("\n⚡ QUICK-FIRE CHEATSHEET ⚡")
            print("   - Replace missing with -1?            -> ❌ FALSE")
            print("   - ML uses Deductive reasoning?        -> ❌ FALSE")
            print("   - Unsupervised has labels?            -> ❌ FALSE")
            print("   - 'Only works with discrete'?         -> ❌ FALSE")
            print("   - Convex = Unique?                    -> ❌ FALSE")
            print("   - Inference: Classify then Extract?   -> ❌ FALSE")
            print("   - Feature Extraction BEFORE predict?  -> ✅ TRUE")
            print("   - ML uses Inductive reasoning?        -> ✅ TRUE")

        elif cheat_choice == 'B':
            print("   - Example: University admissions (Gender bias reversed by Dept choice).")

            print("\n2. DATA WRANGLING vs DATA CLEANING")
            print("   - Hierarchy: Wrangling ⊃ Cleaning.")
            print("   - WRANGLING (Umbrella): The entire process (Collection -> Cleaning -> Reshaping).")
            print("   - CLEANING (Sub-step): Removing nulls, duplicates, outliers.")
            print("   - Trap: They are NOT synonyms. Cleaning is just one part of Wrangling.")

            print("\n3. k-NEAREST NEIGHBOURS (kNN)")
            print("   - Multi-class?: YES, natively. Uses Majority Vote.")
            print("   - No modification needed for >2 classes.")
            print("   - For Regression?: Yes, take MEAN of neighbours instead of vote.")
            print("   - Param k: Small k = Overfit (High Var). Large k = Underfit (High Bias).")

            print("\n4. ANSCOMBE'S QUARTET (The 'Visualise First' Lesson)")
            print("   - Fact: 4 datasets with IDENTICAL Mean, Variance, Correlation, Regression Line.")
            print("   - Visuals: 1. Linear, 2. Curved, 3. Outlier, 4. Vertical Cluster.")
            print("   - Lesson: Summary stats are INSUFFICIENT. Always plot data (EDA).")
            print("   - Regression Property: Line always passes through (mean_x, mean_y).")

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
