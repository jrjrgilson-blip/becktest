"""
Backtest — Grade adaptativa por volatilidade (WIN mini indice), versao Streamlit.
Fontes: Stooq (auto) -> Yahoo (auto) -> Upload CSV -> Demonstracao (sintetico).
Extras: filtro de regime, acao ao virar baixa, trava de esticada, custos, seletor de datas.
"""
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest WIN — Grade Adaptativa", layout="wide")


# ----------------------------- Fontes de dados -----------------------------
def _sinteticos(n=520, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.012, n)
    close = 120000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    op = np.r_[close[0], close[:-1]]
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close}, index=idx)


def _anos(p):
    return {"1y": 1, "2y": 2, "5y": 5, "max": 15}.get(p, 2)


@st.cache_data(show_spinner=False)
def baixar_stooq(ticker, periodo):
    from pandas_datareader import data as pdr
    fim = date.today()
    ini = fim - timedelta(days=365 * _anos(periodo) + 10)
    df = pdr.DataReader(ticker, "stooq", ini, fim)
    if df is None or len(df) == 0:
        raise RuntimeError("stooq vazio")
    return df.sort_index()[["Open", "High", "Low", "Close"]].dropna()


@st.cache_data(show_spinner=False)
def baixar_yahoo(ticker, periodo):
    import yfinance as yf
    df = yf.download(ticker, period=periodo, interval="1d", auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yahoo vazio")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def _num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    s = s.astype(str).str.strip()
    v, p = s.str.contains(",").any(), s.str.contains(r"\.").any()
    if v and p:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif v:
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def ler_csv(arquivo):
    raw = pd.read_csv(arquivo, sep=None, engine="python")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    mapa = {"date": "Date", "data": "Date", "datetime": "Date", "open": "Open",
            "abertura": "Open", "high": "High", "máxima": "High", "maxima": "High",
            "max": "High", "low": "Low", "mínima": "Low", "minima": "Low", "min": "Low",
            "close": "Close", "último": "Close", "ultimo": "Close", "fechamento": "Close",
            "price": "Close", "preço": "Close", "preco": "Close"}
    raw = raw.rename(columns={c: mapa[c] for c in raw.columns if c in mapa})
    faltando = [c for c in ["Date", "Open", "High", "Low", "Close"] if c not in raw.columns]
    if faltando:
        raise RuntimeError(f"CSV sem colunas: {faltando}. Lidas: {list(raw.columns)}")
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        raw[c] = _num(raw[c])
    return raw.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")[
        ["Open", "High", "Low", "Close"]]


def obter_dados(fonte, ticker, periodo, arquivo):
    if fonte == "Upload CSV":
        if arquivo is None:
            raise RuntimeError("Selecione um arquivo CSV.")
        return ler_csv(arquivo), "CSV enviado"
    if fonte == "Demonstração (sintético)":
        return _sinteticos(), "SINTÉTICO (não é real)"
    erros = []
    for nome, fn in [("Stooq", baixar_stooq), ("Yahoo", baixar_yahoo)]:
        try:
            return fn(ticker, periodo), nome
        except Exception as e:
            erros.append(f"{nome}: {e}")
    raise RuntimeError("Nenhuma fonte automática funcionou →\n" + "\n".join(erros))


# ----------------------------- Indicadores + Motor -----------------------------
def preparar(df, atr_len, mm_len):
    df = df.copy()
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(atr_len).mean()
    df["MM"] = c.rolling(mm_len).mean()
    df["MM_up"] = df["MM"] > df["MM"].shift(3)      # inclinacao da media (regime)
    return df


def backtest(df, p):
    entries, realizado, trades, equity, custo = [], 0.0, [], [], 0.0
    cc = p["custo"]
    for dt, row in df.iterrows():
        price, a, mm, mmup = row["Close"], row["ATR"], row["MM"], row["MM_up"]
        if np.isnan(a):
            equity.append(realizado); continue
        bull = (not np.isnan(mm)) and price > mm and bool(mmup)

        # --- acao ao virar baixa (regime) ---
        if p["usar_regime"] and not bull and entries:
            if p["acao"] == "Zerar tudo":
                for e in entries:
                    realizado += (price - e) * p["ponto"] - cc; custo += cc
                    trades.append((dt, "SAIDA_REGIME", price, price - e))
                entries = []
            elif p["acao"] == "Reduzir p/ 1" and len(entries) > 1:
                e = entries.pop()
                realizado += (price - e) * p["ponto"] - cc; custo += cc
                trades.append((dt, "REDUZ_REGIME", price, price - e))

        # --- stop catastrofico de carteira ---
        if entries:
            avg = np.mean(entries)
            if (price - avg) * len(entries) * p["ponto"] < -p["stop_pct"] * p["capital"]:
                for e in entries:
                    realizado += (price - e) * p["ponto"] - cc; custo += cc
                    trades.append((dt, "STOP", price, price - e))
                entries = []

        # --- realizacao ---
        while entries and price >= entries[-1] + p["tp_mult"] * a:
            e = entries.pop()
            realizado += (price - e) * p["ponto"] - cc; custo += cc
            trades.append((dt, "VENDA", price, price - e))

        # --- entradas / adds (so em alta se regime ligado) ---
        pos = len(entries)
        libera = bull or not p["usar_regime"]
        if libera:
            if pos == 0:
                esticado = p["ext_guard"] and not np.isnan(mm) and price > mm + p["ext_mult"] * a
                if not esticado:
                    entries.append(price); realizado -= cc; custo += cc
                    trades.append((dt, "COMPRA", price, 0.0))
            elif pos < p["max_ct"] and price <= entries[-1] - p["add_mult"] * a:
                entries.append(price); realizado -= cc; custo += cc
                trades.append((dt, "ADD", price, 0.0))

        aberto = sum((price - e) for e in entries) * p["ponto"]
        equity.append(realizado + aberto)

    eq = pd.Series(equity, index=df.index, name="Capital (R$)")
    tdf = pd.DataFrame(trades, columns=["Data", "Tipo", "Preco", "Pontos"])
    return eq, tdf, realizado, entries, df, custo


# ----------------------------- UI -----------------------------
st.title("📉 Backtest WIN — Grade Adaptativa por Volatilidade")
st.caption("Proxy padrão: ^BVSP. Para o WIN real, use Upload CSV.")

with st.sidebar:
    st.header("Fonte de dados")
    fonte = st.radio("Origem", ["Automático (Stooq → Yahoo)", "Upload CSV",
                                 "Demonstração (sintético)"], index=0)
    arquivo = st.file_uploader("CSV (Date,Open,High,Low,Close)", type=["csv", "txt"]) \
        if fonte == "Upload CSV" else None
    ticker = st.text_input("Ticker (auto)", "^BVSP")
    periodo = st.selectbox("Período (auto)", ["1y", "2y", "5y", "max"], index=2)

    st.header("Janela de datas")
    usar_datas = st.checkbox("Filtrar por datas", False)
    ini = fim = None
    if usar_datas:
        ini = st.date_input("Início", date(2026, 4, 1))
        fim = st.date_input("Fim", date.today())

    st.header("Grade")
    atr_len = st.number_input("ATR (períodos)", 5, 50, 14)
    add_mult = st.slider("Adiciona a cada X × ATR de queda", 0.3, 3.0, 2.0, 0.1)
    tp_mult = st.slider("Realiza a cada X × ATR de alta", 0.3, 3.0, 2.0, 0.1)
    max_ct = st.slider("Máximo de contratos", 1, 8, 4)

    st.header("Filtro de regime")
    usar_regime = st.checkbox("Ligar filtro de regime (média + inclinação)", True)
    mm_len = st.number_input("Média de tendência (regime)", 10, 300, 72)
    acao = st.selectbox("Ao virar baixa", ["Manter", "Reduzir p/ 1", "Zerar tudo"], index=1)
    ext_guard = st.checkbox("Trava de esticada (não iniciar longe da média)", True)
    ext_mult = st.slider("Distância máx. p/ iniciar (× ATR)", 0.5, 5.0, 2.0, 0.5)

    st.header("Custos e risco")
    custo = st.number_input("Custo por contrato/operação (R$)", 0.0, 20.0, 1.5, 0.5)
    ponto = st.number_input("R$ por ponto (WIN=0,20 / IND=1,00)", 0.05, 1.0, 0.20, 0.05)
    capital = st.number_input("Capital de referência (R$)", 1000, 500000, 20000, 1000)
    stop_pct = st.slider("Stop de carteira (% do capital)", 0.05, 0.50, 0.15, 0.01)

    rodar = st.button("▶ Rodar backtest", type="primary", use_container_width=True)

if rodar:
    try:
        fkey = "Automático" if fonte.startswith("Auto") else fonte
        df, rotulo = obter_dados(fkey, ticker, periodo, arquivo)
    except Exception as e:
        st.error(f"Não consegui carregar dados reais.\n\n{e}\n\nUse **Upload CSV** ou tente depois.")
        st.stop()

    dfi = preparar(df, atr_len, mm_len)          # indicadores no histórico INTEIRO
    if usar_datas:                                # recorta só depois (média já aquecida)
        dfi = dfi.loc[str(ini):str(fim)]
        if len(dfi) < 5:
            st.error("Janela de datas muito curta ou sem dados."); st.stop()

    if "SINTÉTICO" in rotulo:
        st.warning("⚠️ Modo DEMONSTRAÇÃO — números NÃO são reais.")
    else:
        st.success(f"Dados reais via **{rotulo}** — {len(dfi)} candles de "
                   f"{dfi.index[0].date()} a {dfi.index[-1].date()}.")

    p = dict(atr_len=atr_len, add_mult=add_mult, tp_mult=tp_mult, max_ct=max_ct,
             usar_regime=usar_regime, acao=acao, ext_guard=ext_guard, ext_mult=ext_mult,
             ponto=ponto, capital=capital, stop_pct=stop_pct, custo=custo)
    eq, tdf, realizado, entries, dfr, custo_total = backtest(dfi, p)

    n_entradas = int(tdf["Tipo"].isin(["COMPRA", "ADD"]).sum())
    saidas = tdf[tdf["Tipo"].isin(["VENDA", "STOP", "SAIDA_REGIME", "REDUZ_REGIME"])]
    n_stops = int((tdf["Tipo"] == "STOP").sum())
    n_regime = int(tdf["Tipo"].isin(["SAIDA_REGIME", "REDUZ_REGIME"]).sum())
    win = (saidas["Pontos"] > 0).mean() * 100 if len(saidas) else 0.0
    mdd = (eq - eq.cummax()).min()
    rd = (eq.iloc[-1] / abs(mdd)) if mdd < 0 else float("inf")
    aberto_final = sum((dfr["Close"].iloc[-1] - e) for e in entries) * ponto

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resultado líquido (m2m)", f"R$ {eq.iloc[-1]:,.0f}")
    c2.metric("Drawdown máximo", f"R$ {mdd:,.0f}")
    c3.metric("Retorno / Drawdown", f"{rd:.2f}")
    c4.metric("Acerto nas saídas", f"{win:.0f}%")
    c1.metric("Realizado (líq.)", f"R$ {realizado:,.0f}")
    c2.metric("Custos totais", f"R$ {custo_total:,.0f}")
    c3.metric("Entradas", n_entradas)
    c4.metric("Stops / saídas de regime", f"{n_stops} / {n_regime}")

    st.subheader("Curva de capital")
    st.line_chart(eq)
    st.subheader("Preço x Média de tendência")
    st.line_chart(dfr[["Close", "MM"]])
    with st.expander(f"Operações ({len(tdf)})"):
        st.dataframe(tdf, use_container_width=True)
else:
    st.info("Configure na barra lateral e clique em **Rodar backtest**.")
