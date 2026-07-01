"""
Backtest — Grade adaptativa por volatilidade (WIN mini indice), versao Streamlit.
Proxy: ^BVSP (indice a vista). Rode com:  streamlit run streamlit_app.py
No Streamlit Cloud a rede funciona e o yfinance baixa os dados reais.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest WIN — Grade Adaptativa", layout="wide")


# ----------------------------- Motor -----------------------------
def _sinteticos(n=500, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.012, n)
    close = 120000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    op = np.r_[close[0], close[:-1]]
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close}, index=idx)


@st.cache_data(show_spinner=False)
def carregar_dados(ticker, periodo):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=periodo, interval="1d",
                         auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError("download vazio")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close"]].dropna(), True
    except Exception:
        return _sinteticos(), False


def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def backtest(df, p):
    df = df.copy()
    df["ATR"] = atr(df, p["atr_len"])
    df["MM"] = df["Close"].rolling(p["mm_longa"]).mean()

    entries, realizado, trades, equity = [], 0.0, [], []

    for dt, row in df.iterrows():
        price, a, mm = row["Close"], row["ATR"], row["MM"]
        if np.isnan(a):
            equity.append(realizado)
            continue

        # stop catastrofico de carteira
        if entries:
            avg = np.mean(entries)
            if (price - avg) * len(entries) * p["ponto"] < -p["stop_pct"] * p["capital"]:
                for e in entries:
                    realizado += (price - e) * p["ponto"]
                    trades.append((dt, "STOP", price, price - e))
                entries = []

        # realizacao
        while entries and price >= entries[-1] + p["tp_mult"] * a:
            e = entries.pop()
            realizado += (price - e) * p["ponto"]
            trades.append((dt, "VENDA", price, price - e))

        pos = len(entries)
        # entrada / adicao
        if pos == 0:
            entries.append(price); trades.append((dt, "COMPRA", price, 0.0))
        elif pos < p["max_ct"] and price <= entries[-1] - p["add_mult"] * a:
            bloqueia = p["use_mm"] and pos >= 2 and (np.isnan(mm) or price < mm)
            if not bloqueia:
                entries.append(price); trades.append((dt, "ADD", price, 0.0))

        aberto = sum((price - e) for e in entries) * p["ponto"]
        equity.append(realizado + aberto)

    eq = pd.Series(equity, index=df.index, name="Capital (R$)")
    tdf = pd.DataFrame(trades, columns=["Data", "Tipo", "Preco", "Pontos"])
    return eq, tdf, realizado, entries, df


# ----------------------------- UI -----------------------------
st.title("📉 Backtest WIN — Grade Adaptativa por Volatilidade")
st.caption("Proxy: ^BVSP (índice à vista). Ajuste os parâmetros e rode.")

with st.sidebar:
    st.header("Parâmetros")
    ticker = st.text_input("Ticker", "^BVSP")
    periodo = st.selectbox("Período", ["1y", "2y", "5y", "max"], index=1)
    atr_len = st.number_input("ATR (períodos)", 5, 50, 14)
    mm_longa = st.number_input("Média longa (tendência)", 20, 300, 200)
    add_mult = st.slider("Adiciona a cada X × ATR de queda", 0.3, 3.0, 1.0, 0.1)
    tp_mult = st.slider("Realiza a cada X × ATR de alta", 0.3, 3.0, 1.0, 0.1)
    max_ct = st.slider("Máximo de contratos", 1, 8, 4)
    use_mm = st.checkbox("Filtro de tendência (MM) p/ 3º+ contrato", True)
    ponto = st.number_input("R$ por ponto (WIN=0,20 / IND=1,00)", 0.05, 1.0, 0.20, 0.05)
    capital = st.number_input("Capital de referência (R$)", 1000, 500000, 20000, 1000)
    stop_pct = st.slider("Stop de carteira (% do capital)", 0.05, 0.50, 0.15, 0.01)
    rodar = st.button("▶ Rodar backtest", type="primary", use_container_width=True)

if rodar:
    df, real = carregar_dados(ticker, periodo)
    if not real:
        st.warning("yfinance indisponível — usando dados sintéticos só para demonstração. "
                   "Rode no Streamlit Cloud ou com internet para dados reais.")

    p = dict(atr_len=atr_len, mm_longa=mm_longa, add_mult=add_mult, tp_mult=tp_mult,
             max_ct=max_ct, use_mm=use_mm, ponto=ponto, capital=capital, stop_pct=stop_pct)
    eq, tdf, realizado, entries, dfr = backtest(df, p)

    n_entradas = int((tdf["Tipo"].isin(["COMPRA", "ADD"])).sum())
    saidas = tdf[tdf["Tipo"].isin(["VENDA", "STOP"])]
    n_stops = int((tdf["Tipo"] == "STOP").sum())
    win = (saidas["Pontos"] > 0).mean() * 100 if len(saidas) else 0.0
    picos = eq.cummax(); mdd = (eq - picos).min()
    aberto_final = sum((dfr["Close"].iloc[-1] - e) for e in entries) * ponto

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resultado total (m2m)", f"R$ {eq.iloc[-1]:,.0f}")
    c2.metric("Drawdown máximo", f"R$ {mdd:,.0f}")
    c3.metric("Entradas", n_entradas)
    c4.metric("Acerto nas saídas", f"{win:.0f}%")
    c1.metric("Realizado", f"R$ {realizado:,.0f}")
    c2.metric("Stops disparados", n_stops)
    c3.metric("Contratos abertos no fim", len(entries))
    c4.metric("P&L aberto", f"R$ {aberto_final:,.0f}")

    st.subheader("Curva de capital")
    st.line_chart(eq)

    st.subheader("Preço x Média longa")
    st.line_chart(dfr[["Close", "MM"]])

    with st.expander(f"Operações ({len(tdf)})"):
        st.dataframe(tdf, use_container_width=True)
else:
    st.info("Configure os parâmetros na barra lateral e clique em **Rodar backtest**.")
