"""
Backtest — Grade adaptativa por volatilidade (WIN mini indice), versao Streamlit.
Fontes de dados: Stooq (auto) -> Yahoo (auto) -> Upload CSV -> Demonstracao (sintetico).
Rode com:  streamlit run streamlit_app.py
"""

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest WIN — Grade Adaptativa", layout="wide")


# ----------------------------- Fontes de dados -----------------------------
def _sinteticos(n=500, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.012, n)
    close = 120000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    op = np.r_[close[0], close[:-1]]
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close}, index=idx)


def _anos(periodo):
    return {"1y": 1, "2y": 2, "5y": 5, "max": 15}.get(periodo, 2)


@st.cache_data(show_spinner=False)
def baixar_stooq(ticker, periodo):
    from pandas_datareader import data as pdr
    fim = date.today()
    ini = fim - timedelta(days=365 * _anos(periodo) + 10)
    df = pdr.DataReader(ticker, "stooq", ini, fim)
    if df is None or len(df) == 0:
        raise RuntimeError("stooq vazio")
    df = df.sort_index()
    return df[["Open", "High", "Low", "Close"]].dropna()


@st.cache_data(show_spinner=False)
def baixar_yahoo(ticker, periodo):
    import yfinance as yf
    df = yf.download(ticker, period=periodo, interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yahoo vazio")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def _num(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    s = series.astype(str).str.strip()
    tem_v = s.str.contains(",").any()
    tem_p = s.str.contains(r"\.").any()
    if tem_v and tem_p:          # formato BR: 172.024,12
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif tem_v:                  # so virgula decimal
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def ler_csv(arquivo):
    raw = pd.read_csv(arquivo, sep=None, engine="python")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    mapa = {
        "date": "Date", "data": "Date", "datetime": "Date",
        "open": "Open", "abertura": "Open",
        "high": "High", "máxima": "High", "maxima": "High", "max": "High",
        "low": "Low", "mínima": "Low", "minima": "Low", "min": "Low",
        "close": "Close", "último": "Close", "ultimo": "Close",
        "fechamento": "Close", "price": "Close", "preço": "Close", "preco": "Close",
    }
    raw = raw.rename(columns={c: mapa[c] for c in raw.columns if c in mapa})
    faltando = [c for c in ["Date", "Open", "High", "Low", "Close"] if c not in raw.columns]
    if faltando:
        raise RuntimeError(f"CSV sem as colunas: {faltando}. Colunas lidas: {list(raw.columns)}")
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        raw[c] = _num(raw[c])
    raw = raw.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")
    return raw[["Open", "High", "Low", "Close"]]


def obter_dados(fonte, ticker, periodo, arquivo):
    """Retorna (df, rotulo_da_fonte). Levanta excecao com msg clara se falhar."""
    if fonte == "Upload CSV":
        if arquivo is None:
            raise RuntimeError("Selecione um arquivo CSV.")
        return ler_csv(arquivo), "CSV enviado"
    if fonte == "Demonstração (sintético)":
        return _sinteticos(), "SINTÉTICO (não é real)"
    # Automático: tenta Stooq e depois Yahoo, acumulando erros
    erros = []
    for nome, fn in [("Stooq", baixar_stooq), ("Yahoo", baixar_yahoo)]:
        try:
            return fn(ticker, periodo), nome
        except Exception as e:
            erros.append(f"{nome}: {e}")
    raise RuntimeError("Nenhuma fonte automática funcionou →\n" + "\n".join(erros))


# ----------------------------- Motor -----------------------------
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
            equity.append(realizado); continue
        if entries:
            avg = np.mean(entries)
            if (price - avg) * len(entries) * p["ponto"] < -p["stop_pct"] * p["capital"]:
                for e in entries:
                    realizado += (price - e) * p["ponto"]
                    trades.append((dt, "STOP", price, price - e))
                entries = []
        while entries and price >= entries[-1] + p["tp_mult"] * a:
            e = entries.pop()
            realizado += (price - e) * p["ponto"]
            trades.append((dt, "VENDA", price, price - e))
        pos = len(entries)
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
st.caption("Proxy padrão: ^BVSP (índice à vista). Para o WIN real, use Upload CSV.")

with st.sidebar:
    st.header("Fonte de dados")
    fonte = st.radio("Origem", ["Automático (Stooq → Yahoo)", "Upload CSV",
                                 "Demonstração (sintético)"], index=0)
    arquivo = None
    if fonte == "Upload CSV":
        arquivo = st.file_uploader("CSV (Date, Open, High, Low, Close)", type=["csv", "txt"])
    ticker = st.text_input("Ticker (auto)", "^BVSP")
    periodo = st.selectbox("Período (auto)", ["1y", "2y", "5y", "max"], index=1)

    st.header("Parâmetros")
    atr_len = st.number_input("ATR (períodos)", 5, 50, 14)
    mm_longa = st.number_input("Média longa (tendência)", 20, 300, 200)
    add_mult = st.slider("Adiciona a cada X × ATR de queda", 0.3, 3.0, 1.0, 0.1)
    tp_mult = st.slider("Realiza a cada X × ATR de alta", 0.3, 3.0, 1.0, 0.1)
    max_ct = st.slider("Máximo de contratos", 1, 8, 4)
    use_mm = st.checkbox("Filtro de tendência (MM) p/ 3º+ contrato", True)
    ponto = st.number_input("R$ por ponto (WIN=0,20 / IND=1,00)", 0.05, 1.0, 0.20, 0.05)
    capital = st.number_input("Capital de referência (R$)", 1000, 500000, 20000, 1000)
    stop_pct = st.slider("Stop de carteira (% do capital)", 0.05, 0.50, 0.15, 0.01)
    fonte_key = "Automático" if fonte.startswith("Auto") else fonte
    rodar = st.button("▶ Rodar backtest", type="primary", use_container_width=True)

if rodar:
    try:
        fkey = "Automático" if fonte.startswith("Auto") else fonte
        df, rotulo = obter_dados(fkey, ticker, periodo, arquivo)
    except Exception as e:
        st.error(f"Não consegui carregar dados reais.\n\n{e}\n\n"
                 "Dica: use **Upload CSV** (exporte do Profit/TradingView/Investing) "
                 "ou tente novamente mais tarde.")
        st.stop()

    if "SINTÉTICO" in rotulo:
        st.warning("⚠️ Modo DEMONSTRAÇÃO com dados sintéticos — os números abaixo NÃO são reais.")
    else:
        st.success(f"Dados reais carregados via **{rotulo}** — "
                   f"{len(df)} candles de {df.index[0].date()} a {df.index[-1].date()}.")

    p = dict(atr_len=atr_len, mm_longa=mm_longa, add_mult=add_mult, tp_mult=tp_mult,
             max_ct=max_ct, use_mm=use_mm, ponto=ponto, capital=capital, stop_pct=stop_pct)
    eq, tdf, realizado, entries, dfr = backtest(df, p)

    n_entradas = int((tdf["Tipo"].isin(["COMPRA", "ADD"])).sum())
    saidas = tdf[tdf["Tipo"].isin(["VENDA", "STOP"])]
    n_stops = int((tdf["Tipo"] == "STOP").sum())
    win = (saidas["Pontos"] > 0).mean() * 100 if len(saidas) else 0.0
    mdd = (eq - eq.cummax()).min()
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
    st.info("Configure na barra lateral e clique em **Rodar backtest**.")
