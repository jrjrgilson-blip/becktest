"""
Backtest — Grade adaptativa por volatilidade (WIN mini indice), versao Streamlit.
Fontes: Stooq (auto) -> Yahoo (auto) -> Upload CSV -> Demonstracao (sintetico).
Extras: filtro de regime, acao ao virar baixa, trava de esticada, custos, seletor de datas.
"""
import io
import json
import os
from datetime import date, datetime, timedelta

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
                repique_ok = True
                if p["modo_reentrada"] == "Repique na média":
                    repique_ok = (not np.isnan(mm)) and price <= mm + p["reentrada_band"] * a
                if not esticado and repique_ok:
                    for _ in range(min(p["ct_inicial"], p["max_ct"])):
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


def niveis(preco, atr_pts, mm_atual, p):
    """Níveis operacionais a partir do preço/ATR atuais e dos parâmetros."""
    ct_ini = min(p.get("ct_inicial", 1), p["max_ct"])
    linhas, precos = [], []
    for _ in range(ct_ini):
        precos.append(preco)
    avg = round(sum(precos) / len(precos))
    linhas.append({"Nível": "1ª entrada", "Comprar a": round(preco), "Ctr": ct_ini,
                   "Posição": ct_ini, "Alvo de venda": round(preco + p["tp_mult"] * atr_pts),
                   "Preço médio": avg})
    for i in range(1, p["max_ct"] - ct_ini + 1):
        c = round(preco - i * p["add_mult"] * atr_pts)
        precos.append(c)
        avg = round(sum(precos) / len(precos))
        linhas.append({"Nível": f"Adição {i}", "Comprar a": c, "Ctr": 1,
                       "Posição": ct_ini + i, "Alvo de venda": round(c + p["tp_mult"] * atr_pts),
                       "Preço médio": avg})
    teto = round(mm_atual + p["ext_mult"] * atr_pts) if mm_atual else None
    avg_max = sum(precos) / len(precos)
    stop = round(avg_max - p["stop_pct"] * p["capital"] / (p["max_ct"] * p["ponto"]))
    return pd.DataFrame(linhas), teto, mm_atual, stop


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
    atr_len = st.number_input("ATR (períodos)", 5, 50, 6)
    add_mult = st.slider("Adiciona a cada X × ATR de queda", 0.3, 4.0, 2.4, 0.1)
    tp_mult = st.slider("Realiza a cada X × ATR de alta", 0.3, 4.0, 3.0, 0.1)
    max_ct = st.slider("Máximo de contratos", 1, 8, 2)
    ct_inicial = st.number_input("Contratos na 1ª entrada", 1, max_ct, 1)

    st.header("Reentrada")
    modo_reentrada = st.selectbox("Modo de (re)entrada", ["Mercado", "Repique na média"], index=0)
    reentrada_band = st.slider("Repique: distância máx. da média (× ATR)", 0.2, 3.0, 1.0, 0.1)

    st.header("Filtro de regime")
    usar_regime = st.checkbox("Ligar filtro de regime (média + inclinação)", True)
    mm_len = st.number_input("Média de tendência (regime)", 10, 300, 21)
    acao = st.selectbox("Ao virar baixa", ["Manter", "Reduzir p/ 1", "Zerar tudo"], index=1)
    ext_guard = st.checkbox("Trava de esticada (não iniciar longe da média)", True)
    ext_mult = st.slider("Distância máx. p/ iniciar (× ATR)", 0.5, 6.0, 4.0, 0.5)

    st.header("Custos e risco")
    custo = st.number_input("Custo por contrato/operação (R$)", 0.0, 20.0, 1.5, 0.5)
    ponto = st.number_input("R$ por ponto (WIN=0,20 / IND=1,00)", 0.05, 1.0, 0.20, 0.05)
    capital = st.number_input("Capital de referência (R$)", 1000, 500000, 20000, 1000)
    stop_pct = st.slider("Stop de carteira (% do capital)", 0.05, 0.50, 0.10, 0.01)

    rodar = st.button("▶ Rodar backtest", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["📊 Backtest", "🎯 Níveis operacionais", "📌 Acompanhamento"])

with tab1:
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
                 ct_inicial=ct_inicial, modo_reentrada=modo_reentrada, reentrada_band=reentrada_band,
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

with tab2:
    st.subheader("Plano operacional — níveis para colocar as ordens")
    st.caption("Puxe do último candle (diário) ou digite manualmente. O ATR é a chave do "
               "tempo gráfico: para 60min/5min etc., digite o ATR daquele tempo.")
    tf = st.selectbox("Tempo gráfico (rótulo)", ["Diário", "60 min", "15 min", "5 min"], index=0)

    for _k, _v in {"preco_in": 172000.0, "atr_in": 2500.0, "mm_in": 174000.0}.items():
        st.session_state.setdefault(_k, _v)

    if st.button("🔄 Puxar do último candle (auto)"):
        try:
            fk = "Automático" if fonte.startswith("Auto") else fonte
            dfa, rot = obter_dados(fk, ticker, periodo, arquivo)
            dfa = preparar(dfa, atr_len, mm_len).dropna()
            ult = dfa.iloc[-1]
            st.session_state.preco_in = float(round(ult["Close"]))
            st.session_state.atr_in = float(round(ult["ATR"]))
            st.session_state.mm_in = float(round(ult["MM"]))
            st.success(f"Puxado via {rot} ({dfa.index[-1].date()}): preço "
                       f"{st.session_state.preco_in:,.0f} · ATR {st.session_state.atr_in:,.0f} "
                       f"· MM{mm_len} {st.session_state.mm_in:,.0f}")
        except Exception as e:
            st.warning(f"Não consegui puxar automático ({e}). Digite manualmente abaixo.")

    d1, d2, d3 = st.columns(3)
    preco_now = d1.number_input("Pontuação atual do índice", 1000.0, 500000.0,
                                step=100.0, key="preco_in")
    atr_now = d2.number_input(f"ATR atual ({tf}) em pontos", 10.0, 50000.0,
                              step=50.0, key="atr_in")
    mm_now = d3.number_input("Média de tendência atual (pts)", 0.0, 500000.0,
                             step=100.0, key="mm_in")

    plv = dict(add_mult=add_mult, tp_mult=tp_mult, max_ct=max_ct, ct_inicial=ct_inicial,
               ext_mult=ext_mult, ponto=ponto, capital=capital, stop_pct=stop_pct)
    grade, teto, reducao, stop_lvl = niveis(preco_now, atr_now, mm_now, plv)

    if teto and preco_now > teto:
        st.error(f"⚠️ ESTICADO: {preco_now:,.0f} está acima do teto de {teto:,.0f} "
                 f"(MM + {ext_mult:g}×ATR). A trava manda NÃO iniciar compra agora.")
    else:
        st.success(f"OK para iniciar: preço abaixo do teto de esticada ({teto:,.0f} pts).")

    st.markdown("**Grade de entradas e alvos** (compra na queda, vende na alta):")
    st.dataframe(grade, use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Linha de redução (regime)", f"{reducao:,.0f} pts",
              help="Se o preço perder essa linha com a média virando pra baixo: reduzir p/ 1 contrato.")
    m2.metric(f"Stop de carteira ({max_ct} ctr)", f"{stop_lvl:,.0f} pts",
              help=f"Onde a perda aberta atinge {stop_pct:.0%} do capital com a posição cheia.")
    m3.metric("Teto de esticada", f"{teto:,.0f} pts" if teto else "—",
              help="Acima disso, não iniciar posição nova.")

    st.caption(f"Risco máx. teórico do plano ≈ do preço médio até o stop, com {max_ct} contratos. "
               "Distâncias em R$ na tabela usam R$ {:.2f}/ponto.".format(ponto))

    # guarda o plano atual (não fixado) para a aba Acompanhamento
    st.session_state["plano_atual"] = {
        "fixado_em": None,
        "preco_ref": float(preco_now),
        "atr": float(atr_now),
        "mm_snapshot": float(mm_now),
        "entry1": int(grade.iloc[0]["Comprar a"]),
        "alvo1": int(grade.iloc[0]["Alvo de venda"]),
        "entry2": int(grade.iloc[1]["Comprar a"]) if len(grade) > 1 else None,
        "alvo2": int(grade.iloc[1]["Alvo de venda"]) if len(grade) > 1 else None,
        "pm2": int(grade.iloc[-1]["Preço médio"]),
        "stop": int(stop_lvl),
        "teto": int(teto) if teto else None,
        "max_ct": int(max_ct),
        "ponto": float(ponto),
    }


PLANO_PATH = "plano_fixo.json"


def salvar_plano(pl):
    try:
        with open(PLANO_PATH, "w") as f:
            json.dump(pl, f)
    except Exception:
        pass


def carregar_plano():
    try:
        with open(PLANO_PATH) as f:
            return json.load(f)
    except Exception:
        return None


with tab3:
    st.subheader("Acompanhamento — plano fixado")
    st.caption("Fixe o plano uma vez; os níveis abaixo ficam congelados até você resetar. "
               "Só o preço e a média do dia mudam, e as cores dizem o que é permitido.")

    if "plano" not in st.session_state:
        st.session_state["plano"] = carregar_plano()

    ca, cb, cc = st.columns(3)
    if ca.button("📌 Fixar plano atual", use_container_width=True):
        pa = st.session_state.get("plano_atual")
        if pa:
            pa = dict(pa)
            pa["fixado_em"] = datetime.now().strftime("%d/%m/%Y %H:%M")
            st.session_state["plano"] = pa
            salvar_plano(pa)
            st.success("Plano fixado.")
        else:
            st.warning("Abra a aba **Níveis** primeiro para gerar um plano.")
    if cc.button("🗑 Resetar", use_container_width=True):
        st.session_state["plano"] = None
        try:
            os.remove(PLANO_PATH)
        except Exception:
            pass
        st.info("Acompanhamento zerado.")
    up = cb.file_uploader("Carregar .json", type=["json"], label_visibility="collapsed")
    if up is not None:
        try:
            st.session_state["plano"] = json.load(up)
            st.success("Plano carregado do arquivo.")
        except Exception as e:
            st.warning(f"Arquivo inválido ({e}).")

    plano = st.session_state.get("plano")
    if not plano:
        st.info("Nenhum plano fixado ainda. Vá em **🎯 Níveis**, ajuste, volte aqui e clique "
                "em **Fixar plano atual**.")
    else:
        st.markdown(f"**Fixado em {plano.get('fixado_em', '—')}** · preço ref "
                    f"{plano['preco_ref']:,.0f} · ATR {plano['atr']:,.0f}")
        linhas = [{"Nível": "1ª entrada", "Preço": plano["entry1"], "Alvo de venda": plano["alvo1"]}]
        if plano.get("entry2"):
            linhas.append({"Nível": "2ª entrada", "Preço": plano["entry2"],
                           "Alvo de venda": plano["alvo2"]})
        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
        s1, s2 = st.columns(2)
        s1.metric("Stop de carteira (fixo)", f"{plano['stop']:,.0f} pts")
        s2.metric("Teto de esticada (fixo)", f"{plano['teto']:,.0f} pts" if plano.get("teto") else "—")
        st.download_button("⬇️ Baixar plano (.json) para não perder", json.dumps(plano),
                           file_name="plano_fixo.json", mime="application/json")

        st.divider()
        st.markdown("**Atualização do dia** — mexa aqui; os níveis acima continuam fixos:")
        u1, u2, u3 = st.columns(3)
        preco_hoje = u1.number_input("Preço hoje", 1000.0, 500000.0,
                                     float(plano["preco_ref"]), 100.0, key="acomp_preco")
        mm_hoje = u2.number_input("Média (MM21) hoje", 0.0, 500000.0,
                                  float(plano["mm_snapshot"]), 100.0, key="acomp_mm")
        mm_sobe = u3.checkbox("Média subindo", True, key="acomp_slope")

        bull = preco_hoje > mm_hoje and mm_sobe

        # --- validação da 2ª entrada ---
        if plano.get("entry2"):
            e2 = plano["entry2"]
            st.markdown("**2ª entrada (preço médio):**")
            if not mm_sobe:
                st.error("🔴 BLOQUEADA — média virando pra baixo. Não fazer preço médio contra a tendência.")
            elif e2 > mm_hoje:
                st.success(f"🟢 VALIDADA — se cair até {e2:,.0f}, ainda está acima da média "
                           f"({mm_hoje:,.0f}). Pode adicionar o 2º contrato.")
            else:
                st.error(f"🔴 BLOQUEADA — em {e2:,.0f} o preço já estaria abaixo da média "
                         f"({mm_hoje:,.0f}) = regime de baixa. Não adicionar.")

        # --- validação de reentrada (após stop / posição zerada) ---
        st.markdown("**Reentrada (se você zerou por stop ou alvo e está de fora):**")
        teto = plano.get("teto")
        if teto and preco_hoje > teto:
            st.warning(f"🟡 ESTICADO — preço acima do teto ({teto:,.0f}). Espere recuar para iniciar.")
        elif not bull:
            motivo = "abaixo da média" if preco_hoje <= mm_hoje else "com média caindo"
            st.error(f"🔴 NÃO reentrar — preço {preco_hoje:,.0f} {motivo} ({mm_hoje:,.0f}). "
                     "Ainda é baixa; espere o preço reconquistar a média com ela subindo.")
        else:
            st.success("🟢 Reentrada VALIDADA — preço acima da média, média subindo e sem esticar. "
                       "Pode montar um plano novo: vá em Níveis, puxe o candle e fixe de novo.")
