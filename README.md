# Complete ML Toolkit (LES Solver)

A comprehensive Python-based Command-Line Interface (CLI) interactive toolkit for exploring Linear Algebra, Probabilities, Machine Learning Principles, and Linear Equation Systems (L.E.S.). The tool functions as an intelligent computational assistant to parse, validate, solve, and visualize complex matrix-related problems.

---

## � Demo & Visualizations
<div align="center">
  <img src="DemoImages/Dashboard.png" alt="Main Dashboard" width="80%">
    <img src="DemoImages/cheatsheet.png" alt="Main Dashboard" width="80%">
  <p><em>Main Menu Interface & Interactive Cheatsheets</em></p>
  
  <img src="DemoImages/MatrixStats.png" alt="Matrix Analysis" width="80%">
  <p><em>Matrix Solvability Analysis & Engine</em></p>
  
  <img src="DemoImages/2dSurfaceVisualisation.png" alt="2D Visualization" width="45%">
  <img src="DemoImages/SurfaceVisualisation.png" alt="3D Surface Visualization" width="45%">
  <p><em>Matplotlib 2D and 3D Model Rendering</em></p>
</div>

---

## �🛠️ Skills Used

*   **Matrix Algebra & Computational Mathematics:** Deep handling of matrix manipulations (Multiplication, Dot Products, Transposes), Left/Right Pseudo-inverses, Rank computing, and Determinants.
*   **Machine Learning (Regression & Classification):** Implementing Ordinary Least Squares (OLS), Ridge Regression (Primal and Dual forms with L2 normalization), and Multi-Class Classification (One-hot encodings / One-vs-All logic).
*   **Combinatorics & Probability Engine:** Managing permutations, combinations, multisets, Stars & Bars, conditional probability, Bayes theorem, and disjoint events.
*   **Distance Metrics & Search:** Calculating Euclidean (L2) and Manhattan (L1) distances with dynamic K-Nearest Neighbors (KNN) logic.
*   **Data Serialization & Parsing:** Transforming user-provided MATLAB-style string representations (`1,2; 3,4`) directly into structured NumPy multi-dimensional arrays.
*   **Advanced Data Visualization:** Creating conditional plotting algorithms using `matplotlib` to render multi-dimensional datasets, regression lines, and 3D mesh surface planes mapping Actual vs. Predicted values.
*   **CLI UX/UI Design:** Using `colorama` to color-code inputs, results, and warnings sequentially across hierarchical application menus.

---

## 📚 Topics Covered

1.  **System of Linear Equations (L.E.S.) Solvability Analysis:** Analyzing and solving Even, Overdetermined, and Underdetermined systems via matrix rank equivalence versus augmented matrices (`[X | y]`).
2.  **Regularization Methods:** Injecting $\lambda$ weights into Ridge Regression optimizations for edge-case numerically unstable matrix inversions. 
3.  **Polynomial Feature Expansion:** Extending features to explicit polynomial degrees and verifying feature permutations algorithmically. 
4.  **Mathematical Theory & Logic Paradigms:** Dedicated interactive cheat sheets explaining NOIR framework definitions, learning paradigms (Deduction vs Induction), data imputation, classification structures, and dimensionality bounds.
5.  **Performance Evaluation:** Granular outputs for overall and per-column Mean Squared Errors (MSE) evaluating the strength of predictive algorithms against validation metrics.

---

## 🏛️ System Architecture

### 1. High-Level Modular Design

```mermaid
graph TD
    A[Main Menu Interface] --> B[Tool 1: Matrix Math]
    A --> C[Tool 2: Matrix Analyser]
    A --> D[Tool 3: Regression / L.E.S. Solver]
    A --> E[Tool 4: Classifiers & Poly]
    A --> F[Tool 5: KNN & Probability]
    A --> G[Tool 6: ML Cheat Sheets]

    B -.-> B1(Dot Products, Transpose, Math)
    C -.-> C1(Determinants, Inv, Rank Nullity)
    D -.-> D1(OLS, Ridge Regression Primal/Dual)
    E -.-> E1(W-Weights, Polynomal Features)
    F -.-> F1(Combinations, Probabilities)
    G -.-> G1(Interactive Documentation)
```

### 2. Regression & Solvability Engine Pipeline
The sequence of mathematical deduction the system follows when evaluating a set of linearly mapped data targets.

```mermaid
sequenceDiagram
    actor User
    participant Parser
    participant Analyzer
    participant Engine
    participant Visualizer

    User->>Parser: Input X matrix & y vectors
    Parser-->>Analyzer: Cleaned NumPy structures
    Analyzer->>Analyzer: Detect dimensions
    Analyzer->>Analyzer: Compare Rank(X) vs Rank(X_tilde)
    
    alt Unique Solution
        Analyzer->>Engine: Run Standard Inverse
    else Infinite Solutions
        Analyzer->>Engine: Run Minimum Norm
    else No Exact Solution
        Analyzer->>Engine: Run Approximate OLS
    end
    
    opt L2 Regularization
        Engine->>Engine: Apply Ridge Penalty
    end
    
    Engine-->>User: Output Predicted Weights (W)
    Engine->>Visualizer: Pass evaluation sets
    Visualizer-->>User: Render 2D / 3D Plane Plot
```

### 3. Data Processing & Input Parsing Logic

```mermaid
flowchart LR
    Start([User Input String]) --> Check1{"Is it standard Python Array?"}
    Check1 -- Yes --> AST["Evaluate via AST literal"]
    Check1 -- No --> MATLAB["Split by ';'"]
    
    MATLAB --> RowIter["Iterate over Rows"]
    RowIter --> ColIter["Split string by ','"]
    ColIter --> Float{"Cast to float()"}
    
    Float -- Success --> BuildArray(("Construct NumPy Grid"))
    Float -- Empty/Invalid --> Warning["Throw Warning & Continue"]
    
    AST --> BuildArray
```