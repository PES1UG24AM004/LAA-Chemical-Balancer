"""
Chemical Equation Balancer — Python/tkinter port
Act 1: Null-space balancing (Ax = 0)
Act 2: Least-squares sensor recovery (Ax ≈ b)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.linalg import null_space
import random

# --- UI Styling Constants ---
BG       = "#0d0f14"
SURFACE  = "#14181f"
SURFACE2 = "#1c2230"
BORDER   = "#2a3348"
BORDER2  = "#3a4a66"
TEAL     = "#00d4aa"
PURPLE   = "#8b7fff"
AMBER    = "#ffb347"
RED      = "#ff6b6b"
GREEN    = "#4dff91"
TEXT     = "#e8edf5"
TEXT2    = "#8a9ab8"
TEXT3    = "#4a5a78"

PAD_X = 32

EXAMPLES = [
    {
        "label": "NH3 + O2",
        "reactants": ["NH3", "O2"],
        "products":  ["NO", "H2O"],
        "elements":  ["N", "H", "O"],
        "counts": [[1,0,1,0], [3,0,0,2], [0,2,1,1]],
    },
    {
        "label": "H2 + O2",
        "reactants": ["H2", "O2"],
        "products":  ["H2O"],
        "elements":  ["H", "O"],
        "counts": [[2,0,2], [0,2,1]],
    },
    {
        "label": "CH4 + O2",
        "reactants": ["CH4", "O2"],
        "products":  ["CO2", "H2O"],
        "elements":  ["C", "H", "O"],
        "counts": [[1,0,1,0], [4,0,0,2], [0,2,2,1]],
    },
    {
        "label": "Photosynthesis",
        "reactants": ["CO2", "H2O"],
        "products":  ["C6H12O6", "O2"],
        "elements":  ["C", "H", "O"],
        "counts": [[1,0,6,0], [0,2,12,0], [2,1,6,2]],
    },
    {
        "label": "Propane Combustion",
        "reactants": ["C3H8", "O2"],
        "products":  ["CO2", "H2O"],
        "elements":  ["C", "H", "O"],
        "counts": [[3,0,1,0], [8,0,0,2], [0,2,2,1]],
    },
    {
        "label": "Acid-Base (H2SO4)",
        "reactants": ["H2SO4", "NaOH"],
        "products":  ["Na2SO4", "H2O"],
        "elements":  ["H", "S", "O", "Na"],
        "counts": [[2,1,0,2], [1,0,1,0], [4,1,4,1], [0,1,2,0]],
    },
]

# =============================================================================
# MATH ENGINE
# =============================================================================

def to_integers(vec):
    """Scales floating-point vectors into minimal whole integers."""
    vec = np.array(vec, dtype=float)
    abs_vec = np.abs(vec)
    
    nonzero = abs_vec[abs_vec > 1e-9]
    if len(nonzero) == 0:
        return [0] * len(vec)
        
    min_v = nonzero.min()
    scaled = vec / min_v
    
    for denom in range(1, 13):
        ints = np.round(scaled * denom).astype(int)
        if np.allclose(ints, scaled * denom, atol=0.03):
            return list(np.abs(ints))
            
    return list(np.abs(np.round(scaled * 12).astype(int)))

def solve_null_space(signed_matrix):
    """Solves Ax = 0 for exact stoichiometric coefficients."""
    A = np.array(signed_matrix, dtype=float)
    ns = null_space(A)
    
    if ns.shape[1] == 0: return None
        
    vec = ns[:, 0]
    if vec[0] < 0: vec = -vec
        
    return to_integers(vec)

def solve_least_squares(A, b):
    """Solves Ax ≈ b for overdetermined, noisy systems."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    
    pred = A @ x
    residual = float(np.linalg.norm(pred - b))
    
    return x, residual

# =============================================================================
# UI HELPERS
# =============================================================================

def styled_btn(parent, text, command, color=TEAL, fg=BG, **kw):
    return tk.Button(parent, text=text, command=command, bg=color, fg=fg, activebackground=color,
                     font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2", padx=12, pady=6, **kw)

def make_card(parent):
    return tk.Frame(parent, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)

# =============================================================================
# MAIN WINDOW: ACT 1 (NULL SPACE)
# =============================================================================

class ChemicalBalancerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chemical Equation Balancer")
        self.configure(bg=BG)
        self.minsize(860, 750)
        self.resizable(True, True)

        self.reactants = []
        self.products  = []
        self.elements  = []
        self.cell_vars = {}
        self.act1_coeffs = None
        self.active_example = None

        self._build_ui()
        self._load_example(0)

    def _build_ui(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        
        # Safely bind mouse wheel to prevent ghost canvas errors
        def _main_scroll(e, c=canvas):
            if c.winfo_exists():
                c.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _main_scroll)

        inner = self.scroll_frame

        hdr = tk.Frame(inner, bg=BG)
        hdr.pack(fill="x", padx=PAD_X, pady=(24, 4))
        tk.Label(hdr, text="● Linear Algebra · UE24MA241B", bg=SURFACE2, fg=TEAL,
                 font=("Courier", 9, "bold"), padx=10, pady=3).pack(anchor="w", pady=(0, 8))
        tk.Label(hdr, text="Chemical Equation Balancer", bg=BG, fg=TEXT, 
                 font=("Helvetica", 22, "bold")).pack(anchor="w")
        tk.Label(hdr, text="Enter your own equation or pick an example  ·  Act 1: null space  ·  Act 2: least squares",
                 bg=BG, fg=TEXT2, font=("Courier", 9)).pack(anchor="w", pady=(4, 0))

        act1_lbl = tk.Frame(inner, bg=BG)
        act1_lbl.pack(fill="x", padx=PAD_X, pady=(18, 12))
        tk.Label(act1_lbl, text=" Act 1 ", bg=SURFACE2, fg=TEAL, font=("Courier", 9, "bold"), padx=8, pady=2).pack(side="left")
        tx = tk.Frame(act1_lbl, bg=BG)
        tx.pack(side="left", padx=10)
        tk.Label(tx, text="Exact Balancing — Null Space", bg=BG, fg=TEXT, font=("Helvetica", 12, "bold")).pack(anchor="w")

        math_panel = tk.Frame(inner, bg=SURFACE2, highlightthickness=1, highlightbackground=BORDER)
        math_panel.pack(fill="x", padx=PAD_X, pady=(0, 16))
        
        tk.Label(math_panel, text="THE MATH: NULL SPACE (Ax = 0)", bg=SURFACE2, fg=AMBER, 
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=16, pady=(12, 4))
        explanation1 = (
            "A = Matrix of atom counts (Rows = Elements, Columns = Molecules). Products are negative.\n"
            "x = The unknown coefficients we need to balance the equation.\n\n"
            "By the Law of Conservation of Mass, the sum of atoms must equal zero (Reactants - Products = 0). "
            "We use scipy.linalg.null_space to find the basis vector 'x' that perfectly satisfies this homogeneous system."
        )
        tk.Label(math_panel, text=explanation1, bg=SURFACE2, fg=TEXT2, font=("Helvetica", 10), 
                 justify="left", wraplength=780).pack(anchor="w", padx=16, pady=(0, 12))

        self._card1 = make_card(inner)
        self._card1.pack(fill="x", padx=PAD_X, pady=6)
        self._build_card1_equation(self._card1)

        self._card2 = make_card(inner)
        self._card2.pack(fill="x", padx=PAD_X, pady=6)
        self._build_card2_elements(self._card2)

        self._card3 = make_card(inner)
        self._card3.pack(fill="x", padx=PAD_X, pady=6)
        self._build_card3_matrix(self._card3)

        self._result_frame = tk.Frame(inner, bg=BG)
        self._result_frame.pack(fill="x", padx=PAD_X, pady=6)

        cta = tk.Frame(inner, bg=SURFACE2, highlightthickness=1, highlightbackground=PURPLE)
        cta.pack(fill="x", padx=PAD_X, pady=(12, 32))
        tk.Label(cta, text="Act 2 — Real-World Sensor Readings", bg=SURFACE2, fg=TEXT, 
                 font=("Helvetica", 13, "bold")).pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(cta, text="Same reaction, noisy measurements. Overdetermined system. Least squares to recover ratios.",
                 bg=SURFACE2, fg=TEXT2, font=("Courier", 9)).pack(anchor="w", padx=18)
        styled_btn(cta, "Open Act 2 →", self._open_act2, color=PURPLE, fg=BG).pack(anchor="w", padx=18, pady=(10, 16))

    def _build_card1_equation(self, card):
        hdr = tk.Frame(card, bg=SURFACE)
        hdr.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(hdr, text="Step 1 — Equation & Molecules", bg=SURFACE, fg=TEXT, font=("Helvetica", 11, "bold")).pack(side="left")

        pills_frame = tk.Frame(card, bg=SURFACE)
        pills_frame.pack(fill="x", padx=18, pady=(0, 8))
        self._example_pills = []
        for i, ex in enumerate(EXAMPLES):
            btn = tk.Button(pills_frame, text=ex["label"], bg=SURFACE2, fg=TEXT2, font=("Courier", 9), relief="flat", 
                            padx=10, pady=3, cursor="hand2", command=lambda idx=i: self._load_example(idx))
            btn.pack(side="left", padx=3)
            self._example_pills.append(btn)

        eq_wrap = tk.Frame(card, bg=SURFACE)
        eq_wrap.pack(fill="x", padx=18, pady=4)
        self._eq_canvas = tk.Canvas(eq_wrap, bg=SURFACE, highlightthickness=0, height=80)
        self._eq_canvas.pack(fill="x")
        self._eq_inner = tk.Frame(self._eq_canvas, bg=SURFACE)
        self._eq_canvas.create_window((0, 0), window=self._eq_inner, anchor="nw")

        btn_row = tk.Frame(card, bg=SURFACE)
        btn_row.pack(anchor="w", padx=18, pady=(8, 14))
        styled_btn(btn_row, "+ Reactant", self._add_reactant, color=SURFACE2, fg=TEXT2).pack(side="left", padx=4)
        styled_btn(btn_row, "+ Product", self._add_product, color=SURFACE2, fg=TEXT2).pack(side="left", padx=4)
        styled_btn(btn_row, "Clear", self._clear_equation, color=SURFACE, fg=TEXT3).pack(side="left", padx=4)

    def _build_card2_elements(self, card):
        hdr = tk.Frame(card, bg=SURFACE)
        hdr.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(hdr, text="Step 2 — Elements (Matrix Rows)", bg=SURFACE, fg=TEXT, font=("Helvetica", 11, "bold")).pack(side="left")

        self._elem_chips_frame = tk.Frame(card, bg=SURFACE)
        self._elem_chips_frame.pack(fill="x", padx=18, pady=(0, 8))

        add_row = tk.Frame(card, bg=SURFACE)
        add_row.pack(anchor="w", padx=18, pady=(0, 14))
        self._new_elem_var = tk.StringVar()
        elem_entry = tk.Entry(add_row, textvariable=self._new_elem_var, width=14, bg=SURFACE2, fg=TEXT, 
                              insertbackground=TEXT, font=("Courier", 11), relief="flat", 
                              highlightthickness=1, highlightbackground=BORDER2, highlightcolor=TEAL)
        elem_entry.pack(side="left", padx=(0, 8))
        elem_entry.bind("<Return>", lambda e: self._add_element())
        styled_btn(add_row, "+ Add Element", self._add_element, color=SURFACE2, fg=TEXT2).pack(side="left")

    def _build_card3_matrix(self, card):
        hdr = tk.Frame(card, bg=SURFACE)
        hdr.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(hdr, text="Step 3 — Atom Count Matrix A", bg=SURFACE, fg=TEXT, font=("Helvetica", 11, "bold")).pack(side="left")
        
        tk.Label(card, text="→ Enter absolute atom counts. Products shown negative in signed matrix.",
                 bg=SURFACE, fg=TEXT3, font=("Courier", 8)).pack(anchor="w", padx=18)

        mat_outer = tk.Frame(card, bg=SURFACE)
        mat_outer.pack(fill="x", padx=18, pady=4)
        self._mat_canvas = tk.Canvas(mat_outer, bg=SURFACE, highlightthickness=0)
        mat_hsb = ttk.Scrollbar(mat_outer, orient="horizontal", command=self._mat_canvas.xview)
        self._mat_canvas.configure(xscrollcommand=mat_hsb.set)
        mat_hsb.pack(side="bottom", fill="x")
        self._mat_canvas.pack(fill="x")
        
        self._mat_inner = tk.Frame(self._mat_canvas, bg=SURFACE)
        self._mat_canvas.create_window((0, 0), window=self._mat_inner, anchor="nw")
        self._mat_inner.bind("<Configure>", lambda e: self._mat_canvas.configure(
            scrollregion=self._mat_canvas.bbox("all"), height=min(self._mat_inner.winfo_reqheight(), 300)))

        btn_row = tk.Frame(card, bg=SURFACE)
        btn_row.pack(anchor="w", padx=18, pady=(10, 14))
        styled_btn(btn_row, "✓  Solve — Find Null Space", self._solve_null_space, color=TEAL, fg=BG).pack(side="left", padx=4)

    def _render_equation_row(self):
        for w in self._eq_inner.winfo_children(): w.destroy()

        def draw_symbol(txt, color=TEXT3):
            tk.Label(self._eq_inner, text=txt, bg=SURFACE, fg=color, font=("Helvetica", 16)).pack(side="left", padx=4)

        for i, sv in enumerate(self.reactants):
            if i > 0: draw_symbol("+", GREEN)
            self._mol_box(self._eq_inner, sv, "reactant", i)

        draw_symbol("→")

        for i, sv in enumerate(self.products):
            if i > 0: draw_symbol("+", RED)
            self._mol_box(self._eq_inner, sv, "product", i)

        self._eq_inner.update_idletasks()
        self._eq_canvas.configure(height=max(80, self._eq_inner.winfo_reqheight()))

    def _mol_box(self, parent, sv, side, idx):
        frame = tk.Frame(parent, bg=SURFACE2, highlightthickness=1, highlightbackground=BORDER)
        frame.pack(side="left", padx=4, pady=4)

        label_text = f"R{idx+1}" if side == "reactant" else f"P{idx+1}"
        tk.Label(frame, text=label_text, bg=SURFACE2, fg=TEXT3, font=("Courier", 8)).pack(pady=(4, 0))

        entry = tk.Entry(frame, textvariable=sv, width=7, bg=SURFACE2, fg=TEAL, insertbackground=TEAL, 
                         font=("Courier", 12, "bold"), relief="flat", justify="center")
        entry.pack(padx=6, pady=2)
        sv.trace_add("write", lambda *a: self._render_matrix())

        def remove():
            if side == "reactant": self.reactants.pop(idx)
            else: self.products.pop(idx)
            self._render_equation_row()
            self._render_matrix()

        tk.Button(frame, text="×", command=remove, bg=SURFACE2, fg=TEXT3, font=("Courier", 10),
                  relief="flat", cursor="hand2").pack(pady=(0, 4))

    def _render_elements(self):
        for w in self._elem_chips_frame.winfo_children(): w.destroy()
            
        for i, el in enumerate(self.elements):
            chip = tk.Frame(self._elem_chips_frame, bg=SURFACE, highlightthickness=1, highlightbackground=AMBER)
            chip.pack(side="left", padx=4, pady=4)
            tk.Label(chip, text=el, bg=SURFACE, fg=AMBER, font=("Courier", 11, "bold")).pack(side="left", padx=(8, 2))
            tk.Button(chip, text="×", command=lambda idx=i: self._remove_element(idx), bg=SURFACE, fg=TEXT3, 
                      font=("Courier", 10), relief="flat", cursor="hand2").pack(side="left", padx=(0, 4))

    def _render_matrix(self):
        for w in self._mat_inner.winfo_children(): w.destroy()
        self.cell_vars.clear()

        all_mols = [sv.get() for sv in self.reactants] + [sv.get() for sv in self.products]
        nR = len(self.reactants)

        if not all_mols or not self.elements:
            tk.Label(self._mat_inner, text="Add molecules and elements to build the matrix.",
                     bg=SURFACE, fg=TEXT3, font=("Courier", 9)).pack(padx=8, pady=8)
            return

        COL_W = 7 

        hdr = tk.Frame(self._mat_inner, bg=SURFACE)
        hdr.pack(anchor="w")
        tk.Label(hdr, text="Element", width=10, bg=SURFACE2, fg=AMBER, font=("Courier", 9, "bold"),
                 relief="flat", anchor="w", highlightthickness=1, highlightbackground=BORDER).pack(side="left")
                 
        for ci, mol in enumerate(all_mols):
            sign = "+" if ci < nR else "−"
            col  = GREEN if ci < nR else RED
            tk.Label(hdr, text=f"{sign} {mol or '?'}", width=COL_W, bg=SURFACE2, fg=col, font=("Courier", 9, "bold"), 
                     anchor="center", relief="flat", highlightthickness=1, highlightbackground=BORDER).pack(side="left")

        for ri, el in enumerate(self.elements):
            row_frame = tk.Frame(self._mat_inner, bg=SURFACE)
            row_frame.pack(anchor="w")
            tk.Label(row_frame, text=el, width=10, bg=SURFACE2, fg=AMBER, font=("Courier", 9, "bold"), 
                     anchor="w", relief="flat", highlightthickness=1, highlightbackground=BORDER).pack(side="left")
                     
            for ci in range(len(all_mols)):
                stored = getattr(self, "_cell_store", {}).get((ri, ci), "0")
                var = tk.StringVar(value=stored)
                self.cell_vars[(ri, ci)] = var
                
                e = tk.Entry(row_frame, textvariable=var, width=COL_W, bg=SURFACE2, fg=GREEN, insertbackground=TEXT, 
                             font=("Courier", 10), relief="flat", justify="center", highlightthickness=1, 
                             highlightbackground=BORDER, highlightcolor=TEAL)
                e.pack(side="left")
                var.trace_add("write", lambda *a, r=ri, c=ci: self._on_cell_change(r, c))

        self._mat_inner.update_idletasks()

    def _on_cell_change(self, r, c):
        if not hasattr(self, "_cell_store"):
            self._cell_store = {}
        var = self.cell_vars.get((r, c))
        if var:
            self._cell_store[(r, c)] = var.get()

    def _add_reactant(self):
        self.reactants.append(tk.StringVar(value=f"R{len(self.reactants)+1}"))
        self._render_equation_row()
        self._render_matrix()

    def _add_product(self):
        self.products.append(tk.StringVar(value=f"P{len(self.products)+1}"))
        self._render_equation_row()
        self._render_matrix()

    def _add_element(self):
        el = self._new_elem_var.get().strip()
        if el and el not in self.elements:
            self.elements.append(el)
            self._new_elem_var.set("")
            self._render_elements()
            self._render_matrix()

    def _remove_element(self, idx):
        self.elements.pop(idx)
        self._render_elements()
        self._render_matrix()

    def _clear_equation(self):
        self.reactants.clear()
        self.products.clear()
        self.elements.clear()
        self._cell_store = {}
        self.cell_vars.clear()
        self.act1_coeffs = None
        self.active_example = None
        
        self._update_pills()
        self._render_equation_row()
        self._render_elements()
        self._render_matrix()
        self._clear_result()

    def _load_example(self, idx):
        ex = EXAMPLES[idx]
        self.active_example = idx
        self._cell_store = {}
        self.cell_vars.clear()

        self.reactants = [tk.StringVar(value=m) for m in ex["reactants"]]
        self.products  = [tk.StringVar(value=m) for m in ex["products"]]
        self.elements  = list(ex["elements"])

        for ri, row in enumerate(ex["counts"]):
            for ci, val in enumerate(row):
                self._cell_store[(ri, ci)] = str(val)

        self._update_pills()
        self._render_equation_row()
        self._render_elements()
        self._render_matrix()
        self._clear_result()

    def _update_pills(self):
        for i, btn in enumerate(self._example_pills):
            if i == self.active_example:
                btn.config(bg=SURFACE2, fg=TEAL, highlightthickness=1, highlightbackground=TEAL, highlightcolor=TEAL)
            else:
                btn.config(bg=SURFACE2, fg=TEXT2, highlightthickness=0)

    def _get_matrix_data(self):
        """Safely parses user input into floats. Returns None if invalid."""
        all_mols_n = len(self.reactants) + len(self.products)
        data = []
        for ri in range(len(self.elements)):
            row = []
            for ci in range(all_mols_n):
                val_str = self.cell_vars.get((ri, ci), tk.StringVar(value="0")).get().strip()
                if not val_str: val_str = "0"
                try:
                    val = float(val_str)
                    if val < 0: raise ValueError # Atom counts cannot be negative
                    row.append(val)
                except ValueError:
                    return None
            data.append(row)
        return data

    def _get_signed_matrix(self, counts):
        nR = len(self.reactants)
        return [[v if ci < nR else -v for ci, v in enumerate(row)] for row in counts]

    def _solve_null_space(self):
        if not self.elements or (not self.reactants and not self.products):
            messagebox.showerror("Error", "Add molecules and elements first.")
            return

        counts = self._get_matrix_data()
        if counts is None:
            messagebox.showerror("Invalid Input", "Matrix values must be positive numbers. Please correct invalid cells.")
            return

        signed = self._get_signed_matrix(counts)
        coeffs = solve_null_space(signed)

        if coeffs is None:
            self._show_result_error("Could not find a null space vector. Check your matrix for impossibilities.")
            return

        self.act1_coeffs = coeffs
        all_mols = [sv.get() for sv in self.reactants] + [sv.get() for sv in self.products]
        
        self._show_result(coeffs, all_mols, len(self.reactants), counts)

    def _show_result(self, coeffs, all_mols, nR, counts):
        self._clear_result()

        def fmt_side(mols, coeff_list):
            return " + ".join([f"{c}{m}" if c != 1 else m for m, c in zip(mols, coeff_list)])

        lhs = fmt_side([sv.get() for sv in self.reactants], coeffs[:nR])
        rhs = fmt_side([sv.get() for sv in self.products],  coeffs[nR:])

        verified = []
        for ri, el in enumerate(self.elements):
            lhs_sum = sum(counts[ri][ci] * coeffs[ci] for ci in range(nR))
            rhs_sum = sum(counts[ri][ci] * coeffs[ci] for ci in range(nR, len(all_mols)))
            ok = abs(lhs_sum - rhs_sum) < 0.5  
            verified.append((el, lhs_sum, rhs_sum, ok))

        balanced = all(v[3] for v in verified)

        panel = tk.Frame(self._result_frame, bg=SURFACE2, highlightthickness=1, highlightbackground=TEAL if balanced else RED)
        panel.pack(fill="x", pady=8)

        tk.Label(panel, text="BALANCED EQUATION", bg=SURFACE2, fg=TEXT3, font=("Courier", 8)).pack(anchor="w", padx=16, pady=(14, 0))
        tk.Label(panel, text=f"{lhs}  →  {rhs}", bg=SURFACE2, fg=TEAL, font=("Courier", 14, "bold"), wraplength=700).pack(anchor="w", padx=16, pady=(2, 8))

        tk.Label(panel, text="ATOM VERIFICATION", bg=SURFACE2, fg=TEXT3, font=("Courier", 8)).pack(anchor="w", padx=16)
        tbl = tk.Frame(panel, bg=SURFACE2)
        tbl.pack(fill="x", padx=16, pady=4)

        for htext, col, anchor in [("Element", AMBER, "w"), ("Left side", TEXT2, "center"), ("Right side", TEXT2, "center"), ("✓", TEXT2, "center")]:
            tk.Label(tbl, text=htext, bg=SURFACE, fg=col, font=("Courier", 9, "bold"), width=12, anchor=anchor,
                     relief="flat", highlightthickness=1, highlightbackground=BORDER).grid(row=0, column=["Element","Left side","Right side","✓"].index(htext), sticky="nsew")

        for ri, (el, ls, rs, ok) in enumerate(verified, start=1):
            ok_col = GREEN if ok else RED
            for ci, (txt, col, anc) in enumerate([(el, AMBER, "w"), (str(int(ls)), TEXT, "center"),
                                                  (str(int(rs)), TEXT, "center"), ("✓" if ok else "✗", ok_col, "center")]):
                tk.Label(tbl, text=txt, bg=SURFACE2 if ri%2==0 else SURFACE, fg=col, font=("Courier", 9), width=12, anchor=anc,
                         relief="flat", highlightthickness=1, highlightbackground=BORDER).grid(row=ri, column=ci, sticky="nsew")

        tk.Label(panel, text="RAW NULL-SPACE VECTOR", bg=SURFACE2, fg=TEXT3, font=("Courier", 8)).pack(anchor="w", padx=16, pady=(10, 0))
        chips_frame = tk.Frame(panel, bg=SURFACE2)
        chips_frame.pack(anchor="w", padx=16, pady=(4, 0))
        
        for mol, c in zip(all_mols, coeffs):
            tk.Label(chips_frame, text=f"{mol}: {c}", bg="#1a1530", fg=PURPLE, font=("Courier", 10),
                     padx=8, pady=2, relief="flat", highlightthickness=1, highlightbackground=PURPLE).pack(side="left", padx=3, pady=4)

    def _show_result_error(self, msg):
        self._clear_result()
        panel = tk.Frame(self._result_frame, bg=SURFACE2, highlightthickness=1, highlightbackground=RED)
        panel.pack(fill="x", pady=8)
        tk.Label(panel, text=msg, bg=SURFACE2, fg=RED, font=("Courier", 10), wraplength=700).pack(padx=16, pady=14)

    def _clear_result(self):
        for w in self._result_frame.winfo_children(): w.destroy()

    def _open_act2(self):
        if not self.reactants and not self.products:
            self._load_example(0)
            
        if self.act1_coeffs is None:
            counts = self._get_matrix_data()
            if counts is None:
                messagebox.showerror("Invalid Input", "Please correct matrix inputs before opening Act 2.")
                return
            self.act1_coeffs = solve_null_space(self._get_signed_matrix(counts))
            if self.act1_coeffs is None:
                messagebox.showerror("Error", "Balance the equation in Act 1 first.")
                return
                
        Act2Window(self)

# =============================================================================
# SECONDARY WINDOW: ACT 2 (LEAST SQUARES)
# =============================================================================

class Act2Window(tk.Toplevel):
    def __init__(self, app: ChemicalBalancerApp):
        super().__init__(app)
        self.app = app
        self.title("Act 2 — Least Squares")
        self.configure(bg=BG)
        self.minsize(860, 680)

        self.readings = []
        self.reading_vars = {}

        self._build_ui()
        self._generate_readings()

    def _build_ui(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        
        # Safely bind mouse wheel to prevent ghost canvas errors
        def _act2_scroll(e, c=canvas):
            if c.winfo_exists():
                c.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _act2_scroll)

        hdr = tk.Frame(self.inner, bg=BG)
        hdr.pack(fill="x", padx=PAD_X, pady=(20, 4))
        tk.Label(hdr, text="● Act 2", bg=SURFACE2, fg=PURPLE, font=("Courier", 9, "bold"), 
                 padx=10, pady=3).pack(anchor="w", pady=(0, 6))
        tk.Label(hdr, text="Real-World Readings — Least Squares", bg=BG, fg=TEXT, font=("Helvetica", 18, "bold")).pack(anchor="w")

        math_panel = tk.Frame(self.inner, bg=SURFACE2, highlightthickness=1, highlightbackground=BORDER)
        math_panel.pack(fill="x", padx=PAD_X, pady=(0, 16))
        
        tk.Label(math_panel, text="THE MATH: LEAST SQUARES (Ax ≈ b)", bg=SURFACE2, fg=AMBER, 
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=16, pady=(12, 4))
        explanation2 = (
            "A = Matrix of noisy sensor readings for each molecule across multiple tests.\n"
            "b = The total sum of the readings for each test.\n\n"
            "Because real-world sensor data contains random noise, a perfectly intersecting solution doesn't exist. "
            "Instead of solving exactly, we find the vector 'x' that minimizes the squared error ||Ax - b||². "
            "numpy.linalg.lstsq solves this computationally by determining the Normal Equations: x = (AᵀA)⁻¹Aᵀb."
        )
        tk.Label(math_panel, text=explanation2, bg=SURFACE2, fg=TEXT2, font=("Helvetica", 10), 
                 justify="left", wraplength=780).pack(anchor="w", padx=16, pady=(0, 12))

        ctrl = tk.Frame(self.inner, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        ctrl.pack(fill="x", padx=PAD_X, pady=6)
        
        ctrl_inner = tk.Frame(ctrl, bg=SURFACE)
        ctrl_inner.pack(anchor="w", padx=16, pady=12)

        tk.Label(ctrl_inner, text="Sensor noise level:", bg=SURFACE, fg=TEXT2, font=("Courier", 10)).pack(side="left", padx=(0, 8))
        self.noise_var = tk.DoubleVar(value=0.15)
        noise_slider = ttk.Scale(ctrl_inner, from_=0, to=0.5, variable=self.noise_var, orient="horizontal", length=160)
        noise_slider.pack(side="left")
        
        self.noise_label = tk.Label(ctrl_inner, text="0.15", bg=SURFACE, fg=TEAL, font=("Courier", 10, "bold"), width=5)
        self.noise_label.pack(side="left", padx=6)
        self.noise_var.trace_add("write", lambda *a: self.noise_label.config(text=f"{self.noise_var.get():.2f}"))

        tk.Label(ctrl_inner, text="  Readings:", bg=SURFACE, fg=TEXT2, font=("Courier", 10)).pack(side="left", padx=(12, 4))
        self.n_readings_var = tk.IntVar(value=8)
        n_sel = ttk.Combobox(ctrl_inner, textvariable=self.n_readings_var, values=[5, 8, 12, 20], width=4, state="readonly")
        n_sel.pack(side="left", padx=4)

        styled_btn(ctrl_inner, "Generate Readings", self._generate_readings, color=SURFACE2, fg=TEXT2).pack(side="left", padx=10)

        grid_card = tk.Frame(self.inner, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        grid_card.pack(fill="x", padx=PAD_X, pady=6)

        ghdr = tk.Frame(grid_card, bg=SURFACE)
        ghdr.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(ghdr, text="Sensor Readings", bg=SURFACE, fg=TEXT, font=("Helvetica", 11, "bold")).pack(side="left")

        mat_wrap = tk.Frame(grid_card, bg=SURFACE)
        mat_wrap.pack(fill="x", padx=16, pady=4)
        self._grid_canvas = tk.Canvas(mat_wrap, bg=SURFACE, highlightthickness=0)
        hsb = ttk.Scrollbar(mat_wrap, orient="horizontal", command=self._grid_canvas.xview)
        self._grid_canvas.configure(xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        self._grid_canvas.pack(fill="x")
        
        self._grid_inner = tk.Frame(self._grid_canvas, bg=SURFACE)
        self._grid_canvas.create_window((0, 0), window=self._grid_inner, anchor="nw")
        self._grid_inner.bind("<Configure>", lambda e: self._grid_canvas.configure(
            scrollregion=self._grid_canvas.bbox("all"), height=min(self._grid_inner.winfo_reqheight() + 20, 350)))

        btn_row = tk.Frame(grid_card, bg=SURFACE)
        btn_row.pack(anchor="w", padx=16, pady=(10, 14))
        styled_btn(btn_row, "✓  Solve — Least Squares", self._solve_ls, color=PURPLE, fg=BG).pack(side="left", padx=4)
        styled_btn(btn_row, "Regenerate noise", self._generate_readings, color=SURFACE, fg=TEXT3).pack(side="left", padx=4)

        self.result_frame = tk.Frame(self.inner, bg=BG)
        self.result_frame.pack(fill="x", padx=PAD_X, pady=(0, 32))

    def _generate_readings(self):
        all_mols = [sv.get() for sv in self.app.reactants] + [sv.get() for sv in self.app.products]
        true_coeffs = self.app.act1_coeffs

        if not true_coeffs or len(true_coeffs) != len(all_mols):
            messagebox.showerror("Error", "Equation structure changed. Close Act 2 and Re-Solve Act 1 first.")
            self.destroy()
            return

        noise   = self.noise_var.get()
        n_reads = self.n_readings_var.get()
        readings = []
        
        for _ in range(n_reads):
            scale = 0.8 + random.random() * 0.4
            row = []
            for tc in true_coeffs:
                base = tc * scale
                n = (random.random() - 0.5) * 2 * noise * tc
                row.append(max(0.1, base + n)) 
            readings.append(row)

        self.readings = readings
        self._render_readings_grid(readings, all_mols)
        for w in self.result_frame.winfo_children(): w.destroy()

    def _render_readings_grid(self, readings, all_mols):
        for w in self._grid_inner.winfo_children(): w.destroy()
        self.reading_vars = {}
        self.total_labels = {}
        COL_W = 9

        hrow = tk.Frame(self._grid_inner, bg=SURFACE)
        hrow.pack(anchor="w")
        tk.Label(hrow, text="Reading", width=10, bg=SURFACE2, fg=TEXT3, font=("Courier", 9, "bold"), anchor="w",
                 highlightthickness=1, highlightbackground=BORDER).pack(side="left")
                 
        for mol in all_mols:
            tk.Label(hrow, text=mol or "?", width=COL_W, bg=SURFACE2, fg=TEAL, font=("Courier", 9, "bold"), anchor="center",
                     highlightthickness=1, highlightbackground=BORDER).pack(side="left")
                     
        tk.Label(hrow, text="Total (b)", width=COL_W, bg=SURFACE2, fg=AMBER, font=("Courier", 9, "bold"), anchor="center",
                 highlightthickness=1, highlightbackground=BORDER).pack(side="left")

        for ri, row in enumerate(readings):
            rf = tk.Frame(self._grid_inner, bg=SURFACE)
            rf.pack(anchor="w")
            bg_row = SURFACE if ri % 2 == 0 else SURFACE2
            
            tk.Label(rf, text=f"Reading {ri+1}", width=10, bg=bg_row, fg=TEXT3, font=("Courier", 9), anchor="w",
                     highlightthickness=1, highlightbackground=BORDER).pack(side="left")
                     
            for ci, val in enumerate(row):
                var = tk.StringVar(value=f"{val:.3f}")
                self.reading_vars[(ri, ci)] = var
                e = tk.Entry(rf, textvariable=var, width=COL_W, bg=bg_row, fg=TEXT, insertbackground=TEXT, font=("Courier", 9),
                             relief="flat", justify="center", highlightthickness=1, highlightbackground=BORDER)
                e.pack(side="left")
                var.trace_add("write", lambda *a, r=ri: self._recalc_total(r))
                
            total = sum(row)
            total_lbl = tk.Label(rf, text=f"{total:.3f}", width=COL_W, bg=bg_row, fg=AMBER, font=("Courier", 9, "bold"), anchor="center",
                                 highlightthickness=1, highlightbackground=BORDER)
            total_lbl.pack(side="left")
            self.total_labels[ri] = total_lbl

    def _recalc_total(self, ri):
        nM = len(self.app.reactants) + len(self.app.products)
        try:
            total = sum(float(self.reading_vars[(ri, ci)].get()) for ci in range(nM) if self.reading_vars.get((ri, ci)))
            if ri in self.total_labels:
                self.total_labels[ri].config(text=f"{total:.3f}")
        except ValueError:
            pass # Ignore mid-typing errors, catch on final solve instead

    def _get_A_b(self):
        """Safely extracts matrices, returning None if validation fails."""
        all_mols_n = len(self.app.reactants) + len(self.app.products)
        A, b = [], []
        for ri in range(len(self.readings)):
            row = []
            for ci in range(all_mols_n):
                val_str = self.reading_vars.get((ri, ci), tk.StringVar(value="0")).get().strip()
                if not val_str: val_str = "0"
                try:
                    row.append(float(val_str))
                except ValueError:
                    return None, None
            A.append(row)
            b.append(sum(row))
        return A, b

    def _solve_ls(self):
        all_mols = [sv.get() for sv in self.app.reactants] + [sv.get() for sv in self.app.products]
        
        if len(self.app.act1_coeffs) != len(all_mols):
            messagebox.showerror("Error", "Equation structure changed. Close Act 2 and Re-Solve Act 1 first.")
            return

        A, b = self._get_A_b()
        if A is None:
            messagebox.showerror("Invalid Input", "All sensor readings must be valid numbers.")
            return

        x, residual = solve_least_squares(A, b)

        x_scaled = x / (np.abs(x[np.abs(x) > 1e-9]).min() if len(x[np.abs(x) > 1e-9]) else 1.0)
        true_coeffs = np.array(self.app.act1_coeffs, dtype=float)
        true_norm = true_coeffs / true_coeffs[true_coeffs > 0].min()

        for w in self.result_frame.winfo_children(): w.destroy()
        panel = tk.Frame(self.result_frame, bg=SURFACE2, highlightthickness=1, highlightbackground=PURPLE)
        panel.pack(fill="x", pady=8)

        tk.Label(panel, text="LEAST SQUARES SOLUTION", bg=SURFACE2, fg=TEXT3, font=("Courier", 8)).pack(anchor="w", padx=16, pady=(14, 0))
        tk.Label(panel, text="Purple bars = recovered ratios  ·  Teal bars = true ratios from Act 1", bg=SURFACE2, fg=TEXT3, font=("Courier", 8)).pack(anchor="w", padx=16, pady=(0, 8))

        max_val = max(np.abs(x_scaled).max(), np.abs(true_norm).max(), 1.0)
        BAR_MAX = 320

        for i, mol in enumerate(all_mols):
            recovered, truth = float(x_scaled[i]), float(true_norm[i])
            err = abs(recovered - truth)
            
            row = tk.Frame(panel, bg=SURFACE2)
            row.pack(fill="x", padx=16, pady=4)
            tk.Label(row, text=mol, width=8, bg=SURFACE2, fg=TEXT, font=("Courier", 10, "bold"), anchor="w").pack(side="left")

            info = tk.Frame(row, bg=SURFACE2)
            info.pack(side="left", fill="x", expand=True)
            tk.Label(info, text=f"Recovered: {recovered:.3f}   True ratio: {truth:.2f}   Error: {err:.3f}", bg=SURFACE2, fg=TEXT3, font=("Courier", 8)).pack(anchor="w")

            bar_f = tk.Frame(info, bg=BORDER, height=10, width=BAR_MAX)
            bar_f.pack(fill="x", pady=(2, 1))
            bar_f.pack_propagate(False)
            tk.Frame(bar_f, bg=PURPLE, height=10, width=int(abs(recovered) / max_val * BAR_MAX)).place(x=0, y=0)

            tbar_f = tk.Frame(info, bg=BORDER, height=7, width=BAR_MAX)
            tbar_f.pack(fill="x", pady=(0, 2))
            tbar_f.pack_propagate(False)
            tk.Frame(tbar_f, bg=TEAL, height=7, width=int(abs(truth) / max_val * BAR_MAX)).place(x=0, y=0)

        tk.Frame(panel, bg=BORDER, height=1).pack(fill="x", padx=16, pady=10)
        tk.Label(panel, text="Normal equation:  x = (AᵀA)⁻¹Aᵀb", bg=SURFACE2, fg=TEXT2, font=("Courier", 9)).pack(anchor="w", padx=16)
        tk.Label(panel, text=f"Raw solution x = [{', '.join(f'{v:.4f}' for v in x)}]", bg=SURFACE2, fg=TEXT3, font=("Courier", 8)).pack(anchor="w", padx=16)
        
        res_col = GREEN if residual < 0.5 else AMBER
        tk.Label(panel, text=f"Residual ‖Ax−b‖ = {residual:.4f}   {'✓ Low noise' if residual < 0.5 else '⚠ High noise'}", bg=SURFACE2, fg=res_col, font=("Courier", 9, "bold")).pack(anchor="w", padx=16, pady=(4, 16))

if __name__ == "__main__":
    app = ChemicalBalancerApp()
    app.mainloop()