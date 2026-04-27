# -*- coding: utf-8 -*-
import sys
import os

# ── PyInstaller frozen-exe support ────────────────────────────────────────────
# When running as a --onefile exe, PyInstaller extracts bundled files to a
# temporary folder stored in sys._MEIPASS.  We must add that folder to
# sys.path so that `import model_pipeline` and `import recommend` work.
if getattr(sys, "frozen", False):
    _BASE = sys._MEIPASS
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))

if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
# ──────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading

# Resolve the bundled dataset that was added via --add-data in build_exe.py.
# Next to the exe at runtime the csv sits beside the executable; inside the
# frozen bundle it lives in _BASE.
def _find_dataset():
    candidates = [
        os.path.join(_BASE, "final_merged_dataset.csv"),
        os.path.join(os.path.dirname(sys.executable)
                     if getattr(sys, "frozen", False)
                     else os.path.abspath("."),
                     "final_merged_dataset.csv"),
        "final_merged_dataset.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]   # return the _BASE path even if missing (shows error later)

DEFAULT_DATA = _find_dataset()

P = {
    "bg":         "#EDF2FB",
    "sidebar":    "#1B3A6B",
    "side_dark":  "#122850",
    "side_sel":   "#2563B0",
    "side_txt":   "#FFFFFF",
    "side_muted": "#7FA8D4",
    "header":     "#1B3A6B",
    "hdr_txt":    "#FFFFFF",
    "hdr_sub":    "#93C5FD",
    "card":       "#FFFFFF",
    "card_bdr":   "#C7D9F0",
    "txt":        "#1A1A2E",
    "txt_sub":    "#4B5563",
    "txt_muted":  "#8899BB",
    "a1":         "#2563B0",
    "a2":         "#059669",
    "a3":         "#D97706",
    "a4":         "#DC2626",
    "a5":         "#7C3AED",
    "row_odd":    "#EEF4FF",
    "row_even":   "#FFFFFF",
    "inp":        "#FFFFFF",
    "status":     "#122850",
    "log_bg":     "#F0F6FF",
    "sep":        "#9BB8D8",
}

FT  = ("Georgia",      16, "bold")
FH  = ("Georgia",      12, "bold")
FS  = ("Trebuchet MS", 10, "bold")
FB  = ("Trebuchet MS", 10)
FSM = ("Trebuchet MS",  8)
FM  = ("Courier New",   9)
FBG = ("Georgia",      22, "bold")
FMD = ("Georgia",      13, "bold")


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("Hybrid CF Recommender System")
        self.configure(bg=P["bg"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h   = min(1320, sw-60), min(860, sh-60)
        self.geometry(str(w)+"x"+str(h)+"+"+str((sw-w)//2)+"+"+str((sh-h)//2))
        self.minsize(1050, 680)

        self._pipe   = None
        self._busy   = False
        self._dpath  = tk.StringVar(value=DEFAULT_DATA)
        self._eps    = tk.DoubleVar(value=1.0)
        self._mode   = tk.StringVar(value="Full Privacy")
        self._topn   = tk.IntVar(value=10)
        self._user   = tk.StringVar()
        self._status = tk.StringVar(value="Ready")

        self._style()
        self._layout()
        self.after(300, self._boot)

    # ------------------------------------------------------------------ style
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",   background=P["bg"])
        s.configure("TLabel",   background=P["bg"],   foreground=P["txt"],     font=FB)
        s.configure("C.TLabel", background=P["card"], foreground=P["txt"],     font=FB)
        s.configure("TCombobox",fieldbackground=P["inp"], background=P["inp"],
                    foreground=P["txt"], font=FB)
        s.map("TCombobox",
              fieldbackground=[("readonly", P["inp"])],
              foreground=[("readonly", P["txt"])])
        s.configure("Treeview", background=P["card"], foreground=P["txt"],
                    fieldbackground=P["card"], font=FB, rowheight=26)
        s.configure("Treeview.Heading", background=P["header"],
                    foreground=P["hdr_txt"], font=FS, relief="flat")
        s.map("Treeview",
              background=[("selected", P["a1"])],
              foreground=[("selected", "#FFFFFF")])
        s.map("Treeview.Heading",
              background=[("active", P["side_dark"])])
        s.configure("TProgressbar", background=P["a2"],
                    troughcolor=P["card_bdr"], borderwidth=0)

    # ---------------------------------------------------------------- helpers
    def _btn(self, parent, text, cmd, bg=None, fg="#FFFFFF", font=None,
             padx=14, pady=7):
        bg  = bg or P["a1"]
        fnt = font or FS
        b   = tk.Button(parent, text=text, command=cmd,
                        bg=bg, fg=fg, font=fnt, relief="flat", bd=0,
                        cursor="hand2",
                        activebackground=P["side_dark"],
                        activeforeground="#FFFFFF",
                        padx=padx, pady=pady)
        b.bind("<Enter>", lambda e, b=b, o=bg: b.configure(bg=P["side_dark"]))
        b.bind("<Leave>", lambda e, b=b, o=bg: b.configure(bg=o))
        return b

    def _card(self, parent, **kw):
        return tk.Frame(parent, bg=P["card"],
                        highlightbackground=P["card_bdr"],
                        highlightthickness=1, **kw)

    def _sep(self, parent):
        tk.Frame(parent, bg=P["sep"], height=1).pack(fill="x", padx=10, pady=8)

    def _tv(self, parent, cols, widths, height=8):
        f   = tk.Frame(parent, bg=P["card"])
        f.pack(fill="both", expand=True)
        tv  = ttk.Treeview(f, columns=cols, show="headings", height=height)
        for c, w in zip(cols, widths):
            tv.heading(c, text=c.replace("_"," ").title())
            tv.column(c, width=w, anchor="center")
        tv.tag_configure("odd",  background=P["row_odd"])
        tv.tag_configure("even", background=P["row_even"])
        vs  = ttk.Scrollbar(f, orient="vertical",   command=tv.yview)
        hs  = ttk.Scrollbar(f, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side="right",  fill="y")
        hs.pack(side="bottom", fill="x")
        tv.pack(fill="both", expand=True)
        return tv

    # ----------------------------------------------------------------- layout
    def _layout(self):
        self._hdr()
        body = tk.Frame(self, bg=P["bg"])
        body.pack(fill="both", expand=True)
        self._sidebar(body)
        self._content = tk.Frame(body, bg=P["bg"])
        self._content.pack(side="left", fill="both", expand=True)
        self._all_tabs()
        self._statusbar()

    def _hdr(self):
        h = tk.Frame(self, bg=P["header"], height=54)
        h.pack(fill="x")
        h.pack_propagate(False)
        tk.Label(h, text="  Hybrid CF Recommender System",
                 bg=P["header"], fg=P["hdr_txt"], font=FT).pack(side="left", pady=10)
        self._hdr_lbl = tk.Label(h, text="", bg=P["header"],
                                  fg=P["hdr_sub"], font=FSM)
        self._hdr_lbl.pack(side="right", padx=16)

    def _sidebar(self, parent):
        sb = tk.Frame(parent, bg=P["sidebar"], width=196)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)
        self._sb = sb

        tk.Label(sb, text="NAVIGATION", bg=P["sidebar"],
                 fg=P["side_muted"], font=FSM).pack(anchor="w", padx=14, pady=(16,4))

        self._navbtns = {}
        for key, label in [("load","  Load Data"),("train","  Train Model"),
                            ("metrics","  Metrics"),("recommend","  Recommend")]:
            b = tk.Button(sb, text=label, anchor="w",
                          bg=P["sidebar"], fg=P["side_txt"],
                          font=FS, relief="flat", bd=0, cursor="hand2",
                          padx=10, pady=10,
                          activebackground=P["side_sel"],
                          activeforeground="#FFFFFF",
                          command=lambda k=key: self._goto(k))
            b.pack(fill="x", padx=6, pady=1)
            self._navbtns[key] = b

        self._sep(sb)

        tk.Label(sb, text="EPSILON (DP)", bg=P["sidebar"],
                 fg=P["side_muted"], font=FSM).pack(anchor="w", padx=14)
        self._eps_lbl = tk.Label(sb, text="e = 1.00",
                                  bg=P["sidebar"], fg="#FFD700",
                                  font=("Georgia", 17, "bold"))
        self._eps_lbl.pack(pady=(2,2))
        tk.Scale(sb, from_=0.1, to=10.0, resolution=0.1, orient="horizontal",
                  variable=self._eps, bg=P["sidebar"], fg=P["side_txt"],
                  troughcolor=P["side_dark"], highlightthickness=0, bd=0,
                  command=self._eps_change).pack(fill="x", padx=10)
        self._priv_lbl = tk.Label(sb, text="HIGH PRIVACY",
                                   bg=P["sidebar"], fg=P["a2"], font=FSM)
        self._priv_lbl.pack(pady=(0,4))

        self._sep(sb)

        tk.Label(sb, text="MODE", bg=P["sidebar"],
                 fg=P["side_muted"], font=FSM).pack(anchor="w", padx=14)
        for m in ("Full Privacy","Non-Private"):
            tk.Radiobutton(sb, text=m, variable=self._mode, value=m,
                           bg=P["sidebar"], fg=P["side_txt"],
                           selectcolor=P["side_sel"],
                           activebackground=P["sidebar"], font=FSM,
                           command=self._mode_change).pack(anchor="w", padx=14)

        self._sep(sb)

        tk.Label(sb, text="TOP-N", bg=P["sidebar"],
                 fg=P["side_muted"], font=FSM).pack(anchor="w", padx=14)
        row = tk.Frame(sb, bg=P["sidebar"])
        row.pack(fill="x", padx=10)
        for n in (5, 10, 20):
            tk.Radiobutton(row, text=str(n), variable=self._topn, value=n,
                           bg=P["sidebar"], fg=P["side_txt"],
                           selectcolor=P["side_sel"],
                           activebackground=P["sidebar"],
                           font=FSM).pack(side="left", padx=6)

        self._sep(sb)

        tk.Label(sb, text="SYSTEM INFO", bg=P["sidebar"],
                 fg=P["side_muted"], font=FSM).pack(anchor="w", padx=14)
        self._sinfo = {}
        for k in ("Users","Items","Interactions","Sparsity"):
            r = tk.Frame(sb, bg=P["sidebar"])
            r.pack(fill="x", padx=10, pady=1)
            tk.Label(r, text=k+":", bg=P["sidebar"],
                     fg=P["side_muted"], font=FSM,
                     width=13, anchor="w").pack(side="left")
            lbl = tk.Label(r, text="--", bg=P["sidebar"],
                           fg="#93C5FD", font=FSM)
            lbl.pack(side="left")
            self._sinfo[k] = lbl

    def _all_tabs(self):
        self._tabs = {}
        self._tabs["load"]      = self._tab_load()
        self._tabs["train"]     = self._tab_train()
        self._tabs["metrics"]   = self._tab_metrics()
        self._tabs["recommend"] = self._tab_recommend()
        for f in self._tabs.values():
            f.place(in_=self._content, x=0, y=0, relwidth=1, relheight=1)

    def _goto(self, key):
        for k, b in self._navbtns.items():
            b.configure(bg=P["side_sel"] if k==key else P["sidebar"],
                        fg=P["side_txt"])
        self._tabs[key].lift()

    def _statusbar(self):
        bar = tk.Frame(self, bg=P["status"], height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=P["a3"], height=2).pack(fill="x", side="top")
        tk.Label(bar, textvariable=self._status,
                 bg=P["status"], fg="#93C5FD", font=FSM).pack(side="left", padx=10)

    def _setstatus(self, msg):
        self._status.set(msg)
        self._hdr_lbl.configure(text=msg)

    def _log(self, msg):
        self._logtxt.configure(state="normal")
        self._logtxt.insert("end", msg+"\n")
        self._logtxt.see("end")
        self._logtxt.configure(state="disabled")

    def _eps_change(self, val=None):
        v = round(self._eps.get(), 2)
        self._eps_lbl.configure(text="e = "+str(v))
        if v <= 1.0:
            self._priv_lbl.configure(text="HIGH PRIVACY", fg=P["a2"])
        elif v <= 4.0:
            self._priv_lbl.configure(text="MED PRIVACY",  fg=P["a3"])
        else:
            self._priv_lbl.configure(text="LOW PRIVACY",  fg=P["a4"])

    def _mode_change(self):
        if self._mode.get() != "Full Privacy":
            self._priv_lbl.configure(text="NON-PRIVATE", fg=P["a4"])

    # --------------------------------------------------------------- load tab
    def _tab_load(self):
        tab = tk.Frame(self._content, bg=P["bg"])
        tk.Label(tab, text="Data Management",
                 bg=P["bg"], fg=P["a1"], font=FT).pack(anchor="w", padx=12, pady=(8,8))

        pc = self._card(tab)
        pc.pack(fill="x", padx=12, pady=(0,8))
        tk.Label(pc, text="Active Dataset", bg=P["card"],
                 fg=P["txt_sub"], font=FS).pack(anchor="w", padx=14, pady=(10,4))
        pr = tk.Frame(pc, bg=P["card"])
        pr.pack(fill="x", padx=14, pady=(0,10))
        tk.Entry(pr, textvariable=self._dpath, bg=P["inp"], fg=P["txt"],
                 font=FM, relief="solid", bd=1,
                 insertbackground=P["txt"]).pack(side="left", fill="x",
                                                  expand=True, ipady=5)
        self._btn(pr, "Browse", self._browse,
                  bg=P["a3"], fg=P["txt"]).pack(side="left", padx=(8,0))
        self._btn(pr, "Load Dataset",
                  self._do_load).pack(side="left", padx=(8,0))
        self._load_info = tk.Label(pc, text="No dataset loaded",
                                    bg=P["card"], fg=P["txt_muted"], font=FSM)
        self._load_info.pack(anchor="w", padx=14, pady=(0,10))

        sc = self._card(tab)
        sc.pack(fill="x", padx=12, pady=(0,8))
        tk.Label(sc, text="Dataset Statistics", bg=P["card"],
                 fg=P["txt_sub"], font=FS).pack(anchor="w", padx=14, pady=(10,6))
        sg = tk.Frame(sc, bg=P["card"])
        sg.pack(fill="x", padx=14, pady=(0,12))
        self._stw = {}
        defs = [("Total Rows","rows",P["a1"]),("Users","users",P["a2"]),
                ("Items","items",P["a3"]),("Categories","cats",P["a4"]),
                ("Sparsity","sparsity",P["a5"]),("Sources","sources",P["a1"])]
        for col,(lbl,key,col_) in enumerate(defs):
            cel = tk.Frame(sg, bg=P["card_bdr"], padx=1, pady=1)
            cel.grid(row=0, column=col, padx=4, sticky="nsew")
            inn = tk.Frame(cel, bg=P["card"])
            inn.pack(fill="both", expand=True, padx=8, pady=8)
            v   = tk.Label(inn, text="--", bg=P["card"], fg=col_, font=FMD)
            v.pack()
            tk.Label(inn, text=lbl, bg=P["card"],
                     fg=P["txt_muted"], font=FSM).pack()
            self._stw[key] = v
        for c in range(6):
            sg.columnconfigure(c, weight=1)

        pvc = self._card(tab)
        pvc.pack(fill="both", expand=True, padx=12, pady=(0,8))
        ph = tk.Frame(pvc, bg=P["header"])
        ph.pack(fill="x")
        tk.Label(ph, text="Data Preview (first 150 rows)",
                 bg=P["header"], fg=P["hdr_txt"],
                 font=FS).pack(side="left", padx=14, pady=7)
        self._pvcount = tk.Label(ph, text="", bg=P["header"],
                                  fg=P["hdr_sub"], font=FSM)
        self._pvcount.pack(side="right", padx=14)
        cols = ("user_id","item_id","category","interaction_score","source")
        self._pvtree = self._tv(pvc, cols, (130,160,180,140,110), height=12)
        return tab

    # -------------------------------------------------------------- train tab
    def _tab_train(self):
        tab = tk.Frame(self._content, bg=P["bg"])
        tk.Label(tab, text="Model Training",
                 bg=P["bg"], fg=P["a1"], font=FT).pack(anchor="w", padx=12, pady=(8,8))

        cc = self._card(tab)
        cc.pack(fill="x", padx=12, pady=(0,8))
        tk.Label(cc, text="SVM Configuration", bg=P["card"],
                 fg=P["txt_sub"], font=FS).pack(anchor="w", padx=14, pady=(10,6))
        cfg = tk.Frame(cc, bg=P["card"])
        cfg.pack(fill="x", padx=14, pady=(0,10))
        for col,(lbl,val) in enumerate([("Kernel","RBF"),("C","1.0"),
                                         ("Gamma","scale"),("Strategy","OvO"),
                                         ("Max Sample","20 000"),("Features","9")]):
            cel = tk.Frame(cfg, bg=P["card_bdr"], padx=1, pady=1)
            cel.grid(row=0, column=col, padx=4, sticky="nsew")
            inn = tk.Frame(cel, bg=P["card"])
            inn.pack(fill="both", expand=True, padx=10, pady=8)
            tk.Label(inn, text=val, bg=P["card"], fg=P["a1"], font=FH).pack()
            tk.Label(inn, text=lbl, bg=P["card"], fg=P["txt_muted"], font=FSM).pack()
        for c in range(6):
            cfg.columnconfigure(c, weight=1)

        br = tk.Frame(cc, bg=P["card"])
        br.pack(fill="x", padx=14, pady=(0,12))
        self._tbtn = self._btn(br, "  Start Training", self._do_train, bg=P["a2"])
        self._tbtn.pack(side="left")
        self._tprog = ttk.Progressbar(br, mode="indeterminate", length=200)
        self._tprog.pack(side="left", padx=12)
        self._tstat = tk.Label(br, text="", bg=P["card"],
                                fg=P["txt_sub"], font=FSM)
        self._tstat.pack(side="left")

        lc = self._card(tab)
        lc.pack(fill="both", expand=True, padx=12, pady=(0,8))
        lh = tk.Frame(lc, bg=P["header"])
        lh.pack(fill="x")
        tk.Label(lh, text="Training Log", bg=P["header"],
                 fg=P["hdr_txt"], font=FS).pack(side="left", padx=14, pady=7)
        lf = tk.Frame(lc, bg=P["card"])
        lf.pack(fill="both", expand=True)
        self._logtxt = tk.Text(lf, bg=P["log_bg"], fg=P["txt"],
                                font=FM, relief="flat", bd=0,
                                state="disabled", wrap="word")
        ls  = ttk.Scrollbar(lf, orient="vertical",   command=self._logtxt.yview)
        lhs = ttk.Scrollbar(lf, orient="horizontal", command=self._logtxt.xview)
        self._logtxt.configure(yscrollcommand=ls.set, xscrollcommand=lhs.set)
        ls.pack(side="right",  fill="y")
        lhs.pack(side="bottom",fill="x")
        self._logtxt.pack(fill="both", expand=True, padx=4, pady=4)
        return tab

    # ------------------------------------------------------------ metrics tab
    def _tab_metrics(self):
        tab = tk.Frame(self._content, bg=P["bg"])
        tk.Label(tab, text="Model Metrics",
                 bg=P["bg"], fg=P["a1"], font=FT).pack(anchor="w", padx=12, pady=(8,8))

        mc = tk.Frame(tab, bg=P["bg"])
        mc.pack(fill="x", padx=12, pady=(0,8))
        self._mw = {}
        mdefs = [("Accuracy","accuracy",P["a1"]),("Precision","precision",P["a2"]),
                 ("Recall","recall",P["a3"]),("F1-Score","f1",P["a4"]),
                 ("Epsilon","epsilon",P["a5"])]
        for col,(lbl,key,c) in enumerate(mdefs):
            cel = self._card(mc)
            cel.grid(row=0, column=col, padx=5, pady=2, sticky="nsew", ipady=10)
            v   = tk.Label(cel, text="--", bg=P["card"], fg=c, font=FBG)
            v.pack(pady=(10,2))
            tk.Label(cel, text=lbl, bg=P["card"],
                     fg=P["txt_sub"], font=FS).pack(pady=(0,10))
            self._mw[key] = v
        for c in range(5):
            mc.columnconfigure(c, weight=1)

        ec = self._card(tab)
        ec.pack(fill="x", padx=12, pady=(0,8))
        eh = tk.Frame(ec, bg=P["header"])
        eh.pack(fill="x")
        tk.Label(eh, text="Epsilon vs Performance Trade-off",
                 bg=P["header"], fg=P["hdr_txt"],
                 font=FS).pack(side="left", padx=14, pady=7)
        self._eps_sweep_status = tk.Label(eh, text="Pending...",
                                           bg=P["header"], fg=P["hdr_sub"], font=FSM)
        self._eps_sweep_status.pack(side="right", padx=14)
        ecols  = ("epsilon","accuracy","precision","recall","f1")
        ewidths= (100, 110, 110, 110, 110)
        self._etree = self._tv(ec, ecols, ewidths, height=7)

        cvc = self._card(tab)
        cvc.pack(fill="x", padx=12, pady=(0,8))
        cvh = tk.Frame(cvc, bg=P["header"])
        cvh.pack(fill="x")
        tk.Label(cvh, text="5-Fold Cross-Validation",
                 bg=P["header"], fg=P["hdr_txt"],
                 font=FS).pack(side="left", padx=14, pady=7)
        self._cv_status = tk.Label(cvh, text="Pending...",
                                    bg=P["header"], fg=P["hdr_sub"], font=FSM)
        self._cv_status.pack(side="right", padx=14)

        cvf = tk.Frame(cvc, bg=P["card"])
        cvf.pack(fill="x", padx=14, pady=(8,12))
        self._cvl = {}
        cv_defs = [("Accuracy","accuracy",P["a1"]),
                   ("F1 Weighted","f1_weighted",P["a2"]),
                   ("Precision Wtd","precision_weighted",P["a3"]),
                   ("Recall Wtd","recall_weighted",P["a4"])]
        for col,(lbl,key,c) in enumerate(cv_defs):
            cel = tk.Frame(cvf, bg=P["card_bdr"], padx=1, pady=1)
            cel.grid(row=0, column=col, padx=6, sticky="nsew")
            inn = tk.Frame(cel, bg=P["card"])
            inn.pack(fill="both", expand=True, padx=10, pady=10)
            mean_lbl = tk.Label(inn, text="--", bg=P["card"], fg=c, font=FMD)
            mean_lbl.pack()
            std_lbl  = tk.Label(inn, text="std: --", bg=P["card"],
                                 fg=P["txt_muted"], font=FSM)
            std_lbl.pack()
            tk.Label(inn, text=lbl, bg=P["card"],
                     fg=P["txt_sub"], font=FSM).pack()
            self._cvl[key] = (mean_lbl, std_lbl)
        for c in range(4):
            cvf.columnconfigure(c, weight=1)

        return tab

    # ---------------------------------------------------------- recommend tab
    def _tab_recommend(self):
        tab = tk.Frame(self._content, bg=P["bg"])
        tk.Label(tab, text="Live Recommendations",
                 bg=P["bg"], fg=P["a1"], font=FT).pack(anchor="w", padx=12, pady=(8,8))

        cc = self._card(tab)
        cc.pack(fill="x", padx=12, pady=(0,8))
        ci = tk.Frame(cc, bg=P["card"])
        ci.pack(fill="x", padx=14, pady=(10,10))
        tk.Label(ci, text="User ID", bg=P["card"],
                 fg=P["txt_sub"], font=FS).grid(row=0, column=0, sticky="w", pady=4)
        self._ucmb = ttk.Combobox(ci, textvariable=self._user,
                                   state="readonly", font=FB, width=28)
        self._ucmb.grid(row=1, column=0, sticky="w", padx=(0,12))
        self._rbtn = self._btn(ci, "  Get Recommendations", self._do_recommend)
        self._rbtn.grid(row=1, column=1, padx=(0,8))
        self._btn(ci, "Clear", self._clear_recs,
                  bg=P["a4"]).grid(row=1, column=2)
        self._rprog = ttk.Progressbar(ci, mode="indeterminate", length=160)
        self._rprog.grid(row=1, column=3, padx=12)

        rc = self._card(tab)
        rc.pack(fill="both", expand=True, padx=12, pady=(0,8))
        rh = tk.Frame(rc, bg=P["header"])
        rh.pack(fill="x")
        tk.Label(rh, text="Recommendation Results",
                 bg=P["header"], fg=P["hdr_txt"],
                 font=FS).pack(side="left", padx=14, pady=7)
        self._rclbl = tk.Label(rh, text="", bg=P["header"],
                                fg=P["hdr_sub"], font=FSM)
        self._rclbl.pack(side="right", padx=14)
        rcols  = ("#","Item ID","Category","Confidence Score","Avg Interaction","Popularity")
        rwidths= (40, 200, 200, 140, 130, 90)
        self._rtree = self._tv(rc, rcols, rwidths, height=18)
        return tab

    # ------------------------------------------------------------------- boot
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Dataset CSV",
            filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if path:
            self._dpath.set(path)

    def _do_load(self):
        path = self._dpath.get().strip()
        if not os.path.exists(path):
            messagebox.showerror("Not Found","File not found:\n"+path)
            return
        self._setstatus("Loading dataset...")
        self._goto("load")
        def worker():
            try:
                import pandas as pd
                df = pd.read_csv(path)
                self.after(0, lambda: self._show_dataset(df, path))
            except Exception as ex:
                err = str(ex)
                self.after(0, lambda: self._setstatus("Load error: "+err))
        threading.Thread(target=worker, daemon=True).start()

    def _show_dataset(self, df, path):
        n_u  = df["user_id"].nunique()  if "user_id"  in df.columns else 0
        n_i  = df["item_id"].nunique()  if "item_id"  in df.columns else 0
        n_c  = df["category"].nunique() if "category" in df.columns else 0
        srcs = df["source"].unique().tolist() if "source" in df.columns else []
        sp   = round(1.0 - len(df)/max(1, n_u*n_i), 4) if n_u and n_i else 0

        self._stw["rows"].configure(     text=str(len(df)))
        self._stw["users"].configure(    text=str(n_u))
        self._stw["items"].configure(    text=str(n_i))
        self._stw["cats"].configure(     text=str(n_c))
        self._stw["sparsity"].configure( text=str(sp))
        self._stw["sources"].configure(  text=str(len(srcs)))
        self._sinfo["Users"].configure(        text=str(n_u))
        self._sinfo["Items"].configure(        text=str(n_i))
        self._sinfo["Interactions"].configure( text=str(len(df)))
        self._sinfo["Sparsity"].configure(     text=str(sp))

        self._load_info.configure(
            text="Loaded: "+os.path.basename(path)+"  ("+str(len(df))+" rows)",
            fg=P["a2"])
        self._pvcount.configure(
            text=str(min(150,len(df)))+" of "+str(len(df))+" rows shown")

        for r in self._pvtree.get_children():
            self._pvtree.delete(r)
        showcols = ["user_id","item_id","category","interaction_score","source"]
        present  = [c for c in showcols if c in df.columns]
        for i,(_, row) in enumerate(df[present].head(150).iterrows()):
            tag = "odd" if i%2 != 0 else "even"
            self._pvtree.insert("","end",
                values=tuple(str(row.get(c,""))[:35] for c in showcols),
                tags=(tag,))
        self._setstatus("Loaded: "+str(len(df))+" rows | "+
                         str(n_u)+" users | "+str(n_i)+" items")

    def _boot(self):
        path = self._dpath.get().strip()
        if path and os.path.exists(path):
            self._do_load()
            self._try_load_model()
        else:
            self._setstatus("No dataset found. Use Load Data to browse.")
            self.deiconify()

    def _try_load_model(self):
        base = os.path.dirname(self._dpath.get())
        for cand in [
            os.path.join(base, "model_artifacts","hybrid_cf_model.pkl"),
            os.path.join("model_artifacts","hybrid_cf_model.pkl"),
        ]:
            if os.path.exists(cand):
                self._setstatus("Loading saved model...")
                def worker(p=cand):
                    try:
                        from model_pipeline import HybridCFPipeline
                        pipe = HybridCFPipeline.load(p)
                        if pipe.df is None:
                            pipe.load_data(self._dpath.get())
                        self.after(0, lambda: self._model_ready(pipe))
                    except Exception as ex:
                        err = str(ex)
                        self.after(0, lambda: self._setstatus("Model load error: "+err))
                        self.after(0, self.deiconify)
                threading.Thread(target=worker, daemon=True).start()
                return
        self._setstatus("No saved model found. Use Train Model tab.")
        self.deiconify()

    def _model_ready(self, pipe):
        self._pipe = pipe
        users = list(pipe.le_user.classes_)[:300]
        self._ucmb["values"] = users
        if users:
            self._ucmb.current(0)
        self._setstatus("Model ready. "+str(len(users))+" users available.")
        self.deiconify()

    # --------------------------------------------------------------- training
    def _do_train(self):
        if self._busy:
            return
        path = self._dpath.get().strip()
        if not os.path.exists(path):
            messagebox.showerror("No Dataset",
                                  "Load a dataset first from the Load Data tab.")
            return
        self._busy = True
        self._tbtn.configure(state="disabled")
        self._tprog.start(10)
        self._tstat.configure(text="Training...")
        self._goto("train")
        self._logtxt.configure(state="normal")
        self._logtxt.delete("1.0","end")
        self._logtxt.configure(state="disabled")

        for r in self._etree.get_children():
            self._etree.delete(r)
        self._eps_sweep_status.configure(text="Running sweep...")
        self._cv_status.configure(text="Running CV...")
        for key,(mean_lbl,std_lbl) in self._cvl.items():
            mean_lbl.configure(text="--")
            std_lbl.configure(text="std: --")

        def worker():
            try:
                self.after(0, lambda: self._log("Starting training..."))
                self.after(0, lambda: self._log("Dataset : "+path))
                self.after(0, lambda: self._log("Epsilon : "+str(round(self._eps.get(),2))))
                import io, contextlib
                from model_pipeline import HybridCFPipeline
                pipe = HybridCFPipeline(epsilon=self._eps.get())
                buf  = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    metrics = pipe.train(path)
                for line in buf.getvalue().splitlines():
                    ln = line
                    self.after(0, lambda l=ln: self._log(l))
                art = os.path.join(os.path.dirname(path),"model_artifacts")
                pipe.save(art)
                self.after(0, lambda: self._log("Saved to: "+art))
                self.after(0, lambda: self._train_done(pipe, metrics))
            except Exception as ex:
                err = str(ex)
                self.after(0, lambda: self._log("ERROR: "+err))
                self.after(0, lambda: self._train_fail(err))

        threading.Thread(target=worker, daemon=True).start()

    def _train_done(self, pipe, metrics):
        self._busy = False
        self._tbtn.configure(state="normal")
        self._tprog.stop()
        self._tstat.configure(text="Complete.")
        self._log("")
        self._log("--- Results ---")
        for k,v in metrics.items():
            self._log("  "+k+": "+(str(round(v,4)) if isinstance(v,float) else str(v)))

        self._model_ready(pipe)
        self._goto("metrics")

        for key in ("accuracy","precision","recall","f1","epsilon"):
            val  = metrics.get(key,"--")
            disp = str(round(val,4)) if isinstance(val,float) else str(val)
            if key in self._mw:
                self._mw[key].configure(text=disp)

        self._setstatus("Training complete. F1="+str(round(metrics.get("f1",0),4)))
        self._run_background_metrics(pipe)

    def _run_background_metrics(self, pipe):
        def worker():
            import numpy as np
            from sklearn.svm import SVC
            from sklearn.model_selection import (train_test_split,
                                                  cross_validate,
                                                  StratifiedKFold)
            from sklearn.metrics import (accuracy_score, f1_score,
                                          precision_score, recall_score)

            try:
                X_all, y_all = pipe._build_features()
                X_sc         = pipe.scaler.transform(X_all)
            except Exception as ex:
                err = str(ex)
                self.after(0, lambda: self._eps_sweep_status.configure(
                    text="Feature build failed: "+err[:60]))
                self.after(0, lambda: self._cv_status.configure(
                    text="Failed"))
                return

            uniq, cnts = np.unique(y_all, return_counts=True)
            strat      = (y_all if cnts.min() >= 2 and
                          int(len(y_all)*0.2/len(uniq)) >= 1 else None)

            row_idx = [0]

            for eps in [0.1, 0.5, 1.0, 2.0, 5.0, 9.99]:
                try:
                    tmp      = type(pipe)(epsilon=eps)
                    tmp.df        = pipe.df
                    tmp.le_user   = pipe.le_user
                    tmp.le_item   = pipe.le_item
                    tmp.le_cat    = pipe.le_cat
                    tmp.mat       = pipe.mat
                    tmp.user_sim  = pipe.user_sim
                    tmp.item_sim  = pipe.item_sim
                    tmp.scaler    = pipe.scaler
                    tmp.unique_items = pipe.unique_items
                    tmp.items_meta   = pipe.items_meta
                    tmp.svm_model    = pipe.svm_model

                    X2, y2 = tmp._build_features()
                    Xs2    = pipe.scaler.transform(X2)
                    st2    = (y2 if np.unique(y2,return_counts=True)[1].min()>=2
                               and int(len(y2)*0.2/len(np.unique(y2)))>=1 else None)
                    _, Xte2, _, yte2 = train_test_split(Xs2, y2, test_size=0.2,
                                                         random_state=42, stratify=st2)
                    yp2 = pipe.svm_model.predict(Xte2)
                    row_data = (
                        str(eps),
                        str(round(accuracy_score(yte2, yp2), 4)),
                        str(round(precision_score(yte2, yp2, average="weighted",
                                                   zero_division=0), 4)),
                        str(round(recall_score(yte2, yp2, average="weighted",
                                               zero_division=0), 4)),
                        str(round(f1_score(yte2, yp2, average="weighted",
                                           zero_division=0), 4)),
                    )
                    idx = row_idx[0]
                    row_idx[0] += 1
                    self.after(0, lambda r=row_data, i=idx: self._add_eps_row(r, i))
                    self.after(0, lambda e=str(eps): self._eps_sweep_status.configure(
                        text="Completed eps="+e))
                except Exception as ex:
                    err2 = str(ex)
                    self.after(0, lambda e=str(eps), err=err2:
                               self._eps_sweep_status.configure(
                                   text="eps="+e+" failed: "+err[:40]))

            self.after(0, lambda: self._eps_sweep_status.configure(
                text="Sweep complete (" + str(row_idx[0]) + " rows)"))

            try:
                cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cvr = cross_validate(
                    pipe.svm_model, X_sc, y_all, cv=cv,
                    scoring=["accuracy","f1_weighted",
                              "precision_weighted","recall_weighted"])
                self.after(0, lambda: self._show_cv(cvr))
            except Exception as ex:
                err = str(ex)
                self.after(0, lambda: self._cv_status.configure(
                    text="CV failed: "+err[:60]))

        threading.Thread(target=worker, daemon=True).start()

    def _add_eps_row(self, row_data, idx):
        tag = "odd" if idx % 2 != 0 else "even"
        self._etree.insert("","end", values=row_data, tags=(tag,))

    def _show_cv(self, cvr):
        import numpy as np
        for key,(mean_lbl, std_lbl) in self._cvl.items():
            k2 = "test_"+key
            if k2 in cvr:
                mean = round(float(np.mean(cvr[k2])), 4)
                std  = round(float(np.std(cvr[k2])),  4)
                mean_lbl.configure(text=str(mean))
                std_lbl.configure( text="std: "+str(std))
        self._cv_status.configure(text="5-fold CV complete")

    def _train_fail(self, err):
        self._busy = False
        self._tbtn.configure(state="normal")
        self._tprog.stop()
        self._tstat.configure(text="Failed.")
        self._setstatus("Training error: "+err[:80])

    # ----------------------------------------------------------- recommend
    def _do_recommend(self):
        if self._busy:
            return
        if self._pipe is None:
            messagebox.showwarning("No Model","Train or load a model first.")
            return
        uid = self._user.get().strip()
        if not uid:
            messagebox.showwarning("No User","Select a User ID.")
            return
        self._busy = True
        self._rbtn.configure(state="disabled")
        self._rprog.start(10)
        self._clear_recs()
        self._setstatus("Computing recommendations for: "+uid)

        def worker():
            try:
                pipe     = self._pipe
                eps      = self._eps.get()
                if self._mode.get() != "Full Privacy":
                    eps  = 9.99
                pipe.epsilon = eps
                pipe.n_top   = self._topn.get()
                recs     = pipe.predict_for_user(uid)
                self.after(0, lambda: self._show_recs(recs, uid))
            except Exception as ex:
                err = str(ex)
                self.after(0, lambda: self._rec_fail(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_recs(self, recs, uid):
        self._busy = False
        self._rbtn.configure(state="normal")
        self._rprog.stop()
        for r in self._rtree.get_children():
            self._rtree.delete(r)
        for i, row in recs.iterrows():
            tag = "odd" if i%2 != 0 else "even"
            self._rtree.insert("","end", values=(
                i,
                str(row.get("item_id",""))[:30],
                str(row.get("category",""))[:28],
                str(round(row.get("score",0),4)),
                str(round(row.get("avg_interaction",0),3)),
                str(int(row.get("item_popularity",0))),
            ), tags=(tag,))
        n    = len(recs)
        mode = self._mode.get()
        eps  = round(self._eps.get(),2)
        self._rclbl.configure(
            text=str(n)+" items  |  "+mode+"  |  e="+str(eps))
        self._setstatus("Done: "+str(n)+" recommendations for user "+uid)

    def _rec_fail(self, err):
        self._busy = False
        self._rbtn.configure(state="normal")
        self._rprog.stop()
        self._setstatus("Recommendation error: "+err[:80])
        messagebox.showerror("Error", err)

    def _clear_recs(self):
        for r in self._rtree.get_children():
            self._rtree.delete(r)
        self._rclbl.configure(text="")


if __name__ == "__main__":
    App().mainloop()
