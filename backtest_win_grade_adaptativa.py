"""
Backtest — Grade adaptativa por volatilidade (WIN mini indice) usando ^BVSP como proxy.

Estrategia:
  - Espacamento por ATR (nao por pontos fixos): adiciona 1 contrato a cada
    ADD_MULT * ATR de queda abaixo da ultima entrada.
  - Realiza (vende) o ultimo contrato adicionado a cada TP_MULT * ATR de alta.
  - Escalonamento 1 -> MAX_CONTRACTS.
  - Filtro de tendencia: so permite chegar ao 3o/4o contrato se preco > MM200.
  - Stop catastrofico de CARTEIRA: zera tudo se a perda aberta passar de
    STOP_PCT do capital de referencia.

Rode do seu lado (rede liberada). Se o yfinance falhar, cai em dados sinteticos
so para voce conferir que o motor roda.

Requisitos: pip install yfinance numpy pandas matplotlib
"""

import numpy as np
import pandas as pd

# ----------------------- PARAMETROS (ajuste aqui) -----------------------
TICKER          = "^BVSP"     # indice a vista (proxy do WIN)
PERIODO         = "2y"        # ultimos 2 anos
ATR_LEN         = 14
MM_LONGA        = 200
ADD_MULT        = 1.0         # adiciona contrato a cada 1.0 x ATR de queda
TP_MULT         = 1.0         # realiza a cada 1.0 x ATR de alta
MAX_CONTRACTS   = 4
USE_MM200       = True        # filtro de tendencia para 3o/4o contrato
PONTO_RS        = 0.20        # WIN = R$0,20/ponto (IND cheio seria 1,00)
CAPITAL_REF     = 20000.0     # capital de referencia p/ o stop de carteira
STOP_PCT        = 0.15        # zera tudo se perda aberta > 15% do capital
# ------------------------------------------------------------------------


def carregar_dados():
    try:
        import yfinance as yf
        df = yf.download(TICKER, period=PERIODO, interval="1d",
                         auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError("download vazio")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        print(f"[ok] {len(df)} candles reais de {TICKER}")
        return df
    except Exception as e:
        print(f"[aviso] yfinance indisponivel ({e}); usando dados sinteticos.")
        return _sinteticos()


def _sinteticos(n=500, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.012, n)          # leve drift de alta + vol
    close = 120000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low  = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    op   = np.r_[close[0], close[:-1]]
    idx  = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close}, index=idx)


def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def backtest(df):
    df = df.copy()
    df["ATR"] = atr(df, ATR_LEN)
    df["MM"]  = df["Close"].rolling(MM_LONGA).mean()

    entries = []          # precos das entradas abertas (ordem = mais antiga -> mais nova/baixa)
    realizado = 0.0
    trades = []           # (tipo, preco, pontos_resultado)
    equity = []

    for _, row in df.iterrows():
        price, a, mm = row["Close"], row["ATR"], row["MM"]
        if np.isnan(a):
            equity.append(realizado)
            continue

        pos = len(entries)

        # --- stop catastrofico de carteira ---
        if pos > 0:
            avg = np.mean(entries)
            aberto_rs = (price - avg) * pos * PONTO_RS
            if aberto_rs < -STOP_PCT * CAPITAL_REF:
                for e in entries:
                    realizado += (price - e) * PONTO_RS
                    trades.append(("STOP", price, price - e))
                entries = []
                pos = 0

        # --- realizacao (vende ultimo adicionado a +TP_MULT*ATR) ---
        while entries and price >= entries[-1] + TP_MULT * a:
            e = entries.pop()
            realizado += (price - e) * PONTO_RS
            trades.append(("VENDA", price, price - e))

        pos = len(entries)

        # --- entrada / adicao ---
        if pos == 0:
            entries.append(price)
            trades.append(("COMPRA", price, 0.0))
        elif pos < MAX_CONTRACTS and price <= entries[-1] - ADD_MULT * a:
            bloqueia = USE_MM200 and pos >= 2 and (np.isnan(mm) or price < mm)
            if not bloqueia:
                entries.append(price)
                trades.append(("ADD", price, 0.0))

        # equity marcada a mercado
        aberto = sum((price - e) for e in entries) * PONTO_RS
        equity.append(realizado + aberto)

    eq = pd.Series(equity, index=df.index)
    return eq, trades, realizado, entries, df


def stats(eq, trades, realizado, entries, df):
    picos = eq.cummax()
    dd = (eq - picos)
    mdd = dd.min()
    vendas = [t for t in trades if t[0] in ("VENDA", "STOP")]
    ganhos = [t for t in vendas if t[2] > 0]
    win = (len(ganhos) / len(vendas) * 100) if vendas else 0.0
    aberto_final = sum((df["Close"].iloc[-1] - e) for e in entries) * PONTO_RS

    print("\n================ RESULTADO ================")
    print(f"Entradas (compras+adds): {sum(1 for t in trades if t[0] in ('COMPRA','ADD'))}")
    print(f"Realizacoes (vendas+stops): {len(vendas)}  |  stops: {sum(1 for t in trades if t[0]=='STOP')}")
    print(f"Taxa de acerto nas saidas: {win:.1f}%")
    print(f"Contratos abertos no fim: {len(entries)}  (P&L aberto: R$ {aberto_final:,.2f})")
    print(f"Resultado REALIZADO:  R$ {realizado:,.2f}")
    print(f"Resultado TOTAL (m2m): R$ {eq.iloc[-1]:,.2f}")
    print(f"Drawdown maximo:      R$ {mdd:,.2f}")
    print("===========================================")


if __name__ == "__main__":
    df = carregar_dados()
    eq, trades, realizado, entries, df = backtest(df)
    stats(eq, trades, realizado, entries, df)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eq.plot(title="Curva de capital (R$) — grade adaptativa", figsize=(10, 4))
        plt.tight_layout(); plt.savefig("curva_capital.png", dpi=120)
        print("[ok] grafico salvo em curva_capital.png")
    except Exception as e:
        print(f"[info] grafico nao gerado ({e})")
