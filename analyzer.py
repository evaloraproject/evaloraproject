"""
Crypto Futures Signal Bot
Analisa futuros em múltiplas exchanges e envia sinais para Telegram
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import asyncio
import logging
import os
from datetime import datetime
from telegram import Bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Configuração ────────────────────────────────────────────────────────────

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Timeframe principal de análise
TIMEFRAME = "4h"

# Número máximo de pares por exchange
MAX_PAIRS_PER_EXCHANGE = 40

# Mínimo de volume diário em USDT para considerar o par
MIN_VOLUME_USDT = 1_000_000

# Alavancagem sugerida por força de sinal
LEVERAGE_MAP = {
    "forte": 10,
    "médio": 5,
    "fraco":  3,
}

# Alocação % do capital sugerida
ALLOCATION_MAP = {
    "forte": 5,
    "médio": 3,
    "fraco":  2,
}

# Exchanges a monitorizar (apenas as que suportam futuros via ccxt)
EXCHANGES = {
    "binance": ccxt.binanceusdm({"enableRateLimit": True}),
    "gate":    ccxt.gate({"enableRateLimit": True}),
    "mexc":    ccxt.mexc({"enableRateLimit": True}),
    "bitget":  ccxt.bitget({"enableRateLimit": True}),
}

# ─── Funções de análise ───────────────────────────────────────────────────────

def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos os indicadores técnicos no DataFrame OHLCV."""
    df.ta.ema(length=9,  append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=55, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stoch(append=True)
    df["volume_ma"] = df["volume"].rolling(20).mean()
    return df


def avaliar_sinal(df: pd.DataFrame):
    """
    Avalia se há sinal de LONG ou SHORT com base nos indicadores.
    Retorna: ("LONG"|"SHORT"|None, força, score)
    """
    if len(df) < 60:
        return None, None, 0

    r = df.iloc[-1]  # última vela fechada
    score_long  = 0
    score_short = 0

    # ── EMAs ──────────────────────────────────────────────────────────────────
    ema9  = r.get("EMA_9")
    ema21 = r.get("EMA_21")
    ema55 = r.get("EMA_55")
    close = r["close"]

    if ema9 and ema21 and ema55:
        if ema9 > ema21 > ema55 and close > ema9:
            score_long += 2
        elif ema9 < ema21 < ema55 and close < ema9:
            score_short += 2

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi = r.get("RSI_14")
    if rsi:
        if 50 < rsi < 70:
            score_long += 1
        elif rsi > 70:
            score_short += 1   # sobrecomprado → potencial reversão
        if 30 < rsi < 50:
            score_short += 1
        elif rsi < 30:
            score_long += 1    # sobrevendido → potencial reversão

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd = r.get("MACD_12_26_9")
    macd_sig = r.get("MACDs_12_26_9")
    macd_hist = r.get("MACDh_12_26_9")
    if macd and macd_sig:
        if macd > macd_sig and macd_hist and macd_hist > 0:
            score_long += 2
        elif macd < macd_sig and macd_hist and macd_hist < 0:
            score_short += 2

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb_upper = r.get("BBU_20_2.0")
    bb_lower = r.get("BBL_20_2.0")
    if bb_upper and bb_lower:
        if close < bb_lower:
            score_long += 1
        elif close > bb_upper:
            score_short += 1

    # ── ADX (força da tendência) ───────────────────────────────────────────────
    adx = r.get("ADX_14")
    if adx and adx > 25:
        score_long  += 1
        score_short += 1  # tendência forte em qualquer direção

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch_k = r.get("STOCHk_14_3_3")
    stoch_d = r.get("STOCHd_14_3_3")
    if stoch_k and stoch_d:
        if stoch_k < 20 and stoch_k > stoch_d:
            score_long += 1
        elif stoch_k > 80 and stoch_k < stoch_d:
            score_short += 1

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_ma = r.get("volume_ma")
    if vol_ma and r["volume"] > vol_ma * 1.5:
        score_long  += 1
        score_short += 1  # volume confirma qualquer sinal

    # ── Decisão ───────────────────────────────────────────────────────────────
    total = max(score_long, score_short)
    if total < 5:
        return None, None, total

    if score_long > score_short:
        direcao = "LONG"
        score = score_long
    else:
        direcao = "SHORT"
        score = score_short

    if score >= 8:
        forca = "forte"
    elif score >= 6:
        forca = "médio"
    else:
        forca = "fraco"

    return direcao, forca, score


def calcular_zonas(df: pd.DataFrame, direcao: str, preco_atual: float):
    """Calcula entrada, alvos e stop com base em ATR e suportes/resistências."""
    atr = df["ATRr_14"].iloc[-1]
    if not atr or atr == 0:
        atr = preco_atual * 0.015  # fallback 1.5%

    if direcao == "SHORT":
        entrada_min = round(preco_atual * 0.9995, 6)
        entrada_max = round(preco_atual * 1.001,  6)
        alvo1 = round(preco_atual - atr * 1.5, 6)
        alvo2 = round(preco_atual - atr * 3.0, 6)
        alvo3 = round(preco_atual - atr * 5.0, 6)
        stop  = round(preco_atual + atr * 2.0, 6)
    else:
        entrada_min = round(preco_atual * 0.999,  6)
        entrada_max = round(preco_atual * 1.0005, 6)
        alvo1 = round(preco_atual + atr * 1.5, 6)
        alvo2 = round(preco_atual + atr * 3.0, 6)
        alvo3 = round(preco_atual + atr * 5.0, 6)
        stop  = round(preco_atual - atr * 2.0, 6)

    def pct(alvo):
        return round(abs((alvo - preco_atual) / preco_atual) * 100, 2)

    stop_pct = round(abs((stop - preco_atual) / preco_atual) * 100, 2)

    return {
        "entrada_min": entrada_min,
        "entrada_max": entrada_max,
        "alvo1": alvo1, "pct1": pct(alvo1),
        "alvo2": alvo2, "pct2": pct(alvo2),
        "alvo3": alvo3, "pct3": pct(alvo3),
        "stop":  stop,  "stop_pct": stop_pct,
    }


# ─── Formatação da mensagem Telegram ─────────────────────────────────────────

def formatar_mensagem(symbol, direcao, forca, preco, zonas, exchange_name):
    emoji_dir = "💰💰💰" if direcao == "SHORT" else "🚀🚀🚀"
    emoji_seta = "📉" if direcao == "SHORT" else "📈"
    alocacao   = ALLOCATION_MAP[forca]
    alavancagem = LEVERAGE_MAP[forca]
    agora = datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")

    msg = (
        f"{emoji_seta} *{symbol} ({direcao})* {emoji_dir}\n"
        f"⏰ Postado em: {agora}\n"
        f"💲 Preço {exchange_name}: {preco}\n"
        f"⌚ Atualizado em: {agora}\n"
        f"➡️ Preço para entrada: {zonas['entrada_min']} - {zonas['entrada_max']}\n"
        f"➡️ Alocação de patrimônio: {alocacao} %\n"
        f"➡️ Alavancagem: {alavancagem}\n"
        f"🎯 Bons alvos de {'venda' if direcao == 'SHORT' else 'compra'}:\n"
        f"🔜 1ª Zona: {zonas['alvo1']} - {'Vender' if direcao == 'SHORT' else 'Realizar'} 25% (Lucro {zonas['pct1']}%)\n"
        f"🔜 2ª Zona: {zonas['alvo2']} - {'Vender' if direcao == 'SHORT' else 'Realizar'} 25% (Lucro {zonas['pct2']}%)\n"
        f"🔜 3ª Zona: {zonas['alvo3']} - {'Vender' if direcao == 'SHORT' else 'Realizar'} 50% (Lucro {zonas['pct3']}%)\n"
        f"🛑 Stoploss: {zonas['stop']} (-{zonas['stop_pct']}%)\n"
        f"⚠️⚠️⚠️\n"
        f"Nessa operação, utilizaremos o STOP MANUAL. Caso o gráfico confirme o preço de stop, entramos e vendemos tudo manualmente\n"
        f"⚠️⚠️⚠️\n"
        f"⚠️ Força do sinal: *{forca.upper()}* | Exchange: {exchange_name.upper()}"
    )
    return msg


# ─── Loop principal ──────────────────────────────────────────────────────────

async def enviar_telegram(msg: str):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=msg,
        parse_mode="Markdown"
    )


async def analisar_exchange(nome: str, exchange):
    """Analisa todos os pares de futuros de uma exchange e gera sinais."""
    sinais = []
    try:
        log.info(f"[{nome}] A carregar mercados...")
        mercados = exchange.load_markets()

        # Filtrar apenas futuros perpétuos USDT
        pares = [
            s for s, m in mercados.items()
            if "USDT" in s
            and m.get("swap") or m.get("future")
            and m.get("active")
        ]

        # Ordenar por volume e limitar
        pares = pares[:MAX_PAIRS_PER_EXCHANGE]
        log.info(f"[{nome}] Analisando {len(pares)} pares...")

        for symbol in pares:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=120)
                if not ohlcv or len(ohlcv) < 60:
                    continue

                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = calcular_indicadores(df)

                preco_atual = df["close"].iloc[-1]

                # Filtrar por volume mínimo
                vol_usdt = df["volume"].iloc[-1] * preco_atual
                if vol_usdt < MIN_VOLUME_USDT:
                    continue

                direcao, forca, score = avaliar_sinal(df)
                if not direcao:
                    continue

                zonas = calcular_zonas(df, direcao, preco_atual)
                msg   = formatar_mensagem(symbol, direcao, forca, preco_atual, zonas, nome)

                sinais.append({
                    "symbol": symbol,
                    "direcao": direcao,
                    "forca": forca,
                    "score": score,
                    "msg": msg,
                })
                log.info(f"[{nome}] SINAL {direcao} em {symbol} | score={score} | forca={forca}")

            except Exception as e:
                log.debug(f"[{nome}] Erro em {symbol}: {e}")
                continue

    except Exception as e:
        log.error(f"[{nome}] Erro a carregar exchange: {e}")

    return sinais


async def ciclo_analise():
    """Executa um ciclo completo de análise em todas as exchanges."""
    log.info("═══ Iniciando ciclo de análise ═══")
    todos_sinais = []

    tarefas = [analisar_exchange(nome, ex) for nome, ex in EXCHANGES.items()]
    resultados = await asyncio.gather(*tarefas)

    for lista in resultados:
        todos_sinais.extend(lista)

    # Ordenar por score decrescente
    todos_sinais.sort(key=lambda x: x["score"], reverse=True)

    # Enviar apenas os top 5 sinais por ciclo para não spammar
    enviados = 0
    for sinal in todos_sinais:
        if enviados >= 5:
            break
        try:
            await enviar_telegram(sinal["msg"])
            log.info(f"✅ Sinal enviado: {sinal['symbol']} {sinal['direcao']}")
            enviados += 1
            await asyncio.sleep(2)  # pequena pausa entre mensagens
        except Exception as e:
            log.error(f"Erro ao enviar Telegram: {e}")

    if enviados == 0:
        log.info("Nenhum sinal forte encontrado neste ciclo.")

    log.info(f"═══ Ciclo concluído. {enviados} sinais enviados. ═══")


async def main():
    log.info("🤖 Bot de sinais iniciado!")
    # Intervalo entre ciclos (em segundos) — 4h = 14400s
    INTERVALO = 60 * 60 * 4  # a cada 4 horas

    while True:
        await ciclo_analise()
        log.info(f"⏳ Próximo ciclo em {INTERVALO // 60} minutos...")
        await asyncio.sleep(INTERVALO)


if __name__ == "__main__":
    asyncio.run(main())
