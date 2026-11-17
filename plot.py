#!/usr/bin/env python3
"""
Gera as figuras de validação (Figs. 3, 4 e 5 de Boaretto et al., Chaos 33, 123115, 2023)
a partir dos arquivos gravados por neuro.c.

Uso:
    python3 plot.py fig3                  # mapa <F>(g_ext, nu), pontos com eps = 0
    python3 plot.py fig5                  # mapas <R>, CV, <F> em (eps, g_ext), nu = 1
    python3 plot.py fig4                  # rasters de rasters/*.txt (até 9)
    python3 plot.py fig4 arq1.txt arq2.txt
    python3 plot.py todas

Opções:
    --csv resultados.csv     arquivo de resultados (padrão: resultados.csv)
    --saida figuras          pasta de saída       (padrão: figuras)
    --tlim 11000             usa só execuções com esse tlim (padrão: todas)
    --rede barabasi          só esse tipo de rede (dados do neuro_redes.c)
    --N 100                  só esse tamanho de rede (dados do neuro_redes.c)
    --hubs 10                só esse número de hubs estimulados

Com o neuro_redes.c, aponte para o CSV dele:
    python3 plot.py fig5 --csv resultados_redes.csv --rede barabasi

As figuras são salvas em PNG e PDF na pasta de saída.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mapa de cores sequencial. 'plasma' é o mesmo tipo usado no artigo, o que
# facilita a comparação lado a lado na validação.
CMAP = "plasma"
COR_RASTER = "#2a6fdb"
COR_ESTIM = "#d1495b"         # neurônios que recebem o trem de Poisson
COR_NAO_ESTIM = "#3d3d3d"     # neurônios sem entrada externa
T_TRANS = 1000.0          # ms, igual a t_trans em neuro.c
JANELA_RASTER = 500.0     # ms exibidos no raster (como na Fig. 4 do artigo)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ------------------------------------------------------------------ utilidades

def ler_resultados(caminho, tlim=None, rede=None, N=None, hubs=None):
    if not os.path.exists(caminho):
        sys.exit(f"erro: '{caminho}' não encontrado. Rode o neuro antes.")
    df = pd.read_csv(caminho)
    if tlim is not None:
        df = df[np.isclose(df["tlim"], tlim)]
    # CSV do neuro_redes.c: tem as colunas 'rede' e 'N'
    if "rede" in df.columns:
        if rede is not None:
            df = df[df["rede"] == rede]
        elif df["rede"].nunique() > 1:
            sys.exit("erro: o CSV tem mais de um tipo de rede "
                     f"({', '.join(sorted(df['rede'].unique()))}); escolha um com --rede")
    if "hubs" in df.columns:
        if hubs is not None:
            df = df[df["hubs"] == hubs]
        elif df["hubs"].nunique() > 1:
            sys.exit("erro: o CSV tem mais de um número de hubs estimulados "
                     f"({', '.join(map(str, sorted(df['hubs'].unique())))}); escolha um com --hubs")
    if "N" in df.columns:
        if N is not None:
            df = df[df["N"] == N]
        elif df["N"].nunique() > 1:
            sys.exit("erro: o CSV tem mais de um N; escolha um com --N")
    return df


def sufixo(args):
    """Acrescenta rede e N ao nome do arquivo de saída, quando filtrados."""
    s = ""
    if args.rede: s += f"_{args.rede}"
    if args.N:    s += f"_N{args.N}"
    if args.hubs is not None: s += f"_h{args.hubs}"
    return s


def media_por_ponto(df, x, y, colunas):
    """Média (e número de realizações) sobre as seeds para cada par (x, y)."""
    g = df.groupby([y, x])
    m = g[colunas].mean()
    m["n_real"] = g.size()
    return m.reset_index()


def grade(m, x, y, z):
    """Converte a tabela longa em matriz Z[y, x] (NaN onde não há dado)."""
    tab = m.pivot(index=y, columns=x, values=z)
    return tab.columns.values, tab.index.values, tab.values


def bordas_lin(c):
    """Bordas das células para centros c em escala linear."""
    c = np.asarray(c, float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    meio = (c[:-1] + c[1:]) / 2
    return np.concatenate([[c[0] - (meio[0] - c[0])], meio, [c[-1] + (c[-1] - meio[-1])]])


def bordas_log(c):
    """Bordas das células para centros c em escala log."""
    return 10 ** bordas_lin(np.log10(np.asarray(c, float)))


def salvar(fig, pasta, nome):
    os.makedirs(pasta, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(pasta, f"{nome}.{ext}"))
    plt.close(fig)
    print(f"  -> {pasta}/{nome}.png / .pdf")


def avisar_realizacoes(m, rotulo):
    n = m["n_real"]
    print(f"{rotulo}: {len(m)} pontos, realizações por ponto: min {n.min()}, max {n.max()}")


# ----------------------------------------------------------------------- Fig 3

def fig3(args):
    df = ler_resultados(args.csv, args.tlim, args.rede, args.N, args.hubs)
    df = df[df["eps"] == 0]
    if df.empty:
        sys.exit("fig3: nenhuma execução com eps = 0 no CSV.")

    m = media_por_ponto(df, "gExt", "nu", ["F"])
    avisar_realizacoes(m, "fig3")
    gx, nu, F = grade(m, "gExt", "nu", "F")

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    pc = ax.pcolormesh(bordas_lin(gx), bordas_lin(nu), np.ma.masked_invalid(F),
                       cmap=CMAP, vmin=0, shading="flat")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label(r"$\langle F \rangle$ (spikes/s)")
    ax.set_xlabel(r"$g_\mathrm{ext}$ (mS/cm$^2$)")
    ax.set_ylabel(r"$\nu_\mathrm{ext}$ (spikes/ms)")
    ax.set_title("Neurônio isolado: taxa média de disparo", loc="left")
    salvar(fig, args.saida, "fig3" + sufixo(args))


# ----------------------------------------------------------------------- Fig 5

def fig5(args):
    df = ler_resultados(args.csv, args.tlim, args.rede, args.N, args.hubs)
    df = df[(df["eps"] > 0) & np.isclose(df["nu"], 1.0)]
    if df.empty:
        sys.exit("fig5: nenhuma execução com eps > 0 e nu = 1 no CSV.")

    m = media_por_ponto(df, "eps", "gExt", ["R", "CV", "F"])
    avisar_realizacoes(m, "fig5")

    paineis = [
        ("R",  r"$\langle R \rangle$",            (0, 1)),
        ("CV", "CV",                              (0, None)),
        ("F",  r"$\langle F \rangle$ (spikes/s)", (0, None)),
    ]

    fig, axs = plt.subplots(3, 1, figsize=(4.6, 9.6), sharex=True)
    for ax, (col, rot, (vmin, vmax)), letra in zip(axs, paineis, "abc"):
        ex, gy, Z = grade(m, "eps", "gExt", col)
        pc = ax.pcolormesh(bordas_log(ex), bordas_lin(gy), np.ma.masked_invalid(Z),
                           cmap=CMAP, vmin=vmin, vmax=vmax, shading="flat")
        ax.set_xscale("log")
        ax.set_ylabel(r"$g_\mathrm{ext}$ (mS/cm$^2$)")
        cb = fig.colorbar(pc, ax=ax)
        cb.set_label(rot)
        ax.set_title(f"({letra})", loc="left", fontsize=11)
    axs[-1].set_xlabel(r"$\varepsilon$ (mS/cm$^2$)")
    fig.tight_layout()
    salvar(fig, args.saida, "fig5" + sufixo(args))

    # Células sem dado (NaN) aparecem em branco; avisa se houver
    for col in ("R", "CV"):
        n_nan = int(m[col].isna().sum())
        if n_nan:
            print(f"  aviso: {n_nan} ponto(s) com {col} indefinido (poucos spikes) ficam em branco")


# ----------------------------------------------------------------------- Fig 4

def ler_raster(caminho):
    params = {}
    with open(caminho) as f:
        primeira = f.readline()
        for linha in f:
            if not linha.startswith("#"):
                break
            if linha.startswith("# estimulados:"):
                params["estimulados"] = [int(x) for x in linha.split(":", 1)[1].split()]
    if primeira.startswith("#"):
        for item in primeira[1:].split():
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v          # ex.: rede=barabasi
    dados = np.loadtxt(caminho, comments="#", ndmin=2)
    return params, dados


def fig4(args):
    arquivos = args.arquivos or sorted(glob.glob("rasters/*.txt"))
    if not arquivos:
        sys.exit("fig4: nenhum raster encontrado (rode o neuro com o 6º argumento = 1).")
    rasters = [ler_raster(a) for a in arquivos]
    if args.rede:
        rasters = [r for r in rasters if r[0].get("rede", "random") == args.rede]
    if args.N:
        rasters = [r for r in rasters if int(r[0].get("N", 100)) == args.N]
    if args.hubs is not None:
        rasters = [r for r in rasters if int(r[0].get("hubs", -1)) == args.hubs]
    if not rasters:
        sys.exit("fig4: nenhum raster com a rede/N pedidos.")
    if len(rasters) > 9:
        print(f"fig4: {len(rasters)} rasters encontrados; usando os 9 primeiros.")
        rasters = rasters[:9]
    # ordena como no artigo: linhas = g_ext crescente, colunas = eps crescente
    rasters.sort(key=lambda r: (r[0].get("gExt", 0), r[0].get("eps", 0)))

    n = len(rasters)
    ncol = min(3, n)
    nlin = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nlin, ncol, figsize=(3.4 * ncol, 2.6 * nlin),
                            sharex=True, sharey=True, squeeze=False)

    for ax, (p, d) in zip(axs.flat, rasters):
        t0 = p.get("t_trans", T_TRANS)
        if d.size:
            sel = (d[:, 1] >= t0) & (d[:, 1] < t0 + JANELA_RASTER)
            est = set(p.get("estimulados", []))
            Nr = int(p.get("N", 100))
            if est and len(est) < Nr:
                # como na Fig. 8 do artigo: estimulados coloridos, demais em cinza escuro
                m_est = np.array([int(i) in est for i in d[:, 0]])
                for mask, cor, rot in ((sel & ~m_est, COR_NAO_ESTIM, "sem entrada externa"),
                                       (sel & m_est, COR_ESTIM, "estimulados")):
                    ax.plot(d[mask, 1] - t0, d[mask, 0], ".", ms=1.5, color=cor,
                            rasterized=True, label=rot)
            else:
                ax.plot(d[sel, 1] - t0, d[sel, 0], ".", ms=1.5, color=COR_RASTER,
                        rasterized=True)
        ax.set_title(rf"$\varepsilon$ = {p.get('eps', float('nan')):g},  "
                     rf"$g_\mathrm{{ext}}$ = {p.get('gExt', float('nan')):g}", fontsize=9)
        ax.set_xlim(0, JANELA_RASTER)
        ax.set_ylim(-1, p.get("N", 100))
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    h, l = axs.flat[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2, markerscale=6, frameon=False,
                   bbox_to_anchor=(0.5, 1.03))
    for ax in axs[-1, :]:
        ax.set_xlabel("t (ms, após o transiente)")
    for ax in axs[:, 0]:
        ax.set_ylabel("índice do neurônio")
    fig.tight_layout()
    salvar(fig, args.saida, "fig4" + sufixo(args))


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figura", choices=["fig3", "fig4", "fig5", "todas"])
    ap.add_argument("arquivos", nargs="*", help="rasters específicos (só para fig4)")
    ap.add_argument("--csv", default="resultados.csv")
    ap.add_argument("--saida", default="figuras")
    ap.add_argument("--tlim", type=float, default=None)
    ap.add_argument("--rede", choices=["random", "barabasi", "bianconi"], default=None,
                    help="filtra um tipo de rede (CSV/rasters do neuro_redes)")
    ap.add_argument("--N", type=int, default=None, help="filtra um tamanho de rede")
    ap.add_argument("--hubs", type=int, default=None,
                    help="filtra o número de hubs estimulados (redes barabasi/bianconi)")
    args = ap.parse_args()

    if args.figura in ("fig3", "todas"):
        fig3(args)
    if args.figura in ("fig5", "todas"):
        fig5(args)
    if args.figura in ("fig4", "todas"):
        fig4(args)


if __name__ == "__main__":
    main()
