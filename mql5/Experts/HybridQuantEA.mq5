//+------------------------------------------------------------------+
//|                                              HybridQuantEA.mq5    |
//|  Regime-switching EA: mean reversion + trend following            |
//|  Built for prop accounts with equity-based trailing drawdown      |
//+------------------------------------------------------------------+
#property copyright "HybridQuant"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include <HybridQuant/RiskManager.mqh>
#include <HybridQuant/Strategies.mqh>

//--- Account / Risk Parameters --------------------------------------
input group "=== Account & Risk ==="
input double InpInitialBalance   = 5000.0;  // Account size at deployment
input double InpMaxDrawdownPct   = 6.0;     // Total DD allowance (%) - verify with firm rules
input double InpSoftStopRatio    = 0.60;    // Halt entries at this share of DD budget
input double InpRiskMeanRev      = 0.40;    // Risk per mean-reversion trade (% equity)
input double InpRiskTrend        = 0.30;    // Risk per trend trade (% equity)
input double InpRiskCrypto       = 0.20;    // Risk per crypto trade (% equity)
input double InpMaxRoomFraction  = 0.25;    // Max share of remaining DD room per trade
input int    InpMaxOpenPositions = 3;       // Cap on concurrent positions

//--- Symbol Configuration -------------------------------------------
input group "=== Symbols ==="
input string InpFxSymbols        = "EURUSD,XAUUSD"; // Comma-separated FX/metal symbols
input string InpCryptoSymbols    = "BTCUSD";        // Comma-separated crypto symbols
input bool   InpTradeCrypto      = true;

//--- Strategy Parameters --------------------------------------------
input group "=== Strategy ==="
input ENUM_TIMEFRAMES InpEntryTF  = PERIOD_H1;
input ENUM_TIMEFRAMES InpRegimeTF = PERIOD_H4;
input int    InpBBPeriod          = 20;
input double InpBBDeviation       = 2.0;
input int    InpRSIPeriod         = 14;
input double InpRSIOversold       = 30.0;
input double InpRSIOverbought     = 70.0;
input int    InpATRPeriod         = 14;
input int    InpADXPeriod         = 14;
input double InpADXRangeMax       = 20.0;  // ADX below this = ranging
input double InpADXTrendMin       = 25.0;  // ADX above this = trending
input int    InpMAFast            = 20;
input int    InpMASlow            = 50;
input int    InpBreakoutLookback  = 20;

//--- Trade Management & Exits ---------------------------------------
input group "=== Exits & Execution ==="
input double InpSLAtrMult         = 2.0;   // Stop distance = ATR x this
input double InpSLAtrMultCrypto   = 3.0;   // Stop multiplier for crypto assets
input double InpTrendTPRMultiple  = 2.0;   // Trend profit target as R multiple
input double InpPartialCloseR     = 1.0;   // Take partial close at this R
input double InpPartialClosePct   = 50.0;  // Volume percentage closed at partial
input double InpTrailStartR       = 0.5;   // Activate trailing stop after this R
input double InpTrailAtrMult      = 1.5;   // Trailing distance in ATR
input int    InpMaxSpreadPoints   = 35;    // Max permissible spread in points

//--- Compliance & Magic Numbers -------------------------------------
input group "=== Compliance ==="
input int    InpMinHoldSeconds    = 150;   // Minimum hold time in seconds
input int    InpMagicMeanRev      = 770101;
input int    InpMagicTrend        = 770102;
input int    InpMagicCrypto       = 770103;
input int    InpSlippagePoints    = 20;

//--- Engine Globals -------------------------------------------------
CTrade          g_trade;
CRiskManager    g_risk;
CStrategyEngine g_engine;

SymbolContext   g_fx[];
SymbolContext   g_crypto[];
datetime        g_last_bar_time = 0;

//+------------------------------------------------------------------+
//| Helper Prototypes                                                |
//+------------------------------------------------------------------+
int    SplitSymbols(const string csv, SymbolContext &out[]);
bool   IsNewBar(void);
bool   IsWeekend(void);
void   ManageOpenPositions(void);
double ParseRUnit(const string comment);
bool   IsPartialDone(const string comment);
double ATRForSymbol(const string symbol);
bool   RespectsStopLevel(const string symbol, const double price, const double sl);
bool   HasPositionOn(const string symbol, const long magic);
void   TryEnter(SymbolContext &ctx, const ENUM_SIGNAL sig, const long magic,
                const double risk_pct, const double sl_atr_mult, const bool use_tp_multiple);

//+------------------------------------------------------------------+
int SplitSymbols(const string csv, SymbolContext &out[])
{
   string parts[];
   const int n = StringSplit(csv, ',', parts);
   if(n <= 0) return 0;

   ArrayResize(out, n);
   int valid = 0;
   for(int i = 0; i < n; i++)
   {
      string s = parts[i];
      StringTrimLeft(s);
      StringTrimRight(s);
      if(StringLen(s) == 0) continue;

      if(g_engine.InitSymbol(out[valid], s))
         valid++;
   }
   ArrayResize(out, valid);
   return valid;
}

//+------------------------------------------------------------------+
int OnInit()
{
   if(InpInitialBalance <= 0.0)
   {
      Print("[INIT] InpInitialBalance must be positive.");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(InpADXRangeMax >= InpADXTrendMin)
   {
      Print("[INIT] InpADXRangeMax must be strictly below InpADXTrendMin to maintain dead band.");
      return INIT_PARAMETERS_INCORRECT;
   }

   g_engine.Configure(InpEntryTF, InpRegimeTF,
                      InpBBPeriod, InpBBDeviation,
                      InpRSIPeriod, InpRSIOversold, InpRSIOverbought,
                      InpATRPeriod,
                      InpADXPeriod, InpADXRangeMax, InpADXTrendMin,
                      InpMAFast, InpMASlow);

   g_risk.Init(InpInitialBalance, InpMaxDrawdownPct, InpSoftStopRatio);

   g_trade.SetDeviationInPoints(InpSlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   g_trade.LogLevel(LOG_LEVEL_ERRORS);

   const int fx_count = SplitSymbols(InpFxSymbols, g_fx);
   int crypto_count = 0;
   if(InpTradeCrypto)
      crypto_count = SplitSymbols(InpCryptoSymbols, g_crypto);

   if(fx_count == 0 && crypto_count == 0)
   {
      Print("[INIT] Failed to initialize any valid tradable symbols.");
      return INIT_FAILED;
   }

   PrintFormat("[INIT] HybridQuantEA initialized. Active FX: %d, Crypto: %d. Drawdown budget: %.2f",
               fx_count, crypto_count, InpInitialBalance * InpMaxDrawdownPct / 100.0);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   for(int i = 0; i < ArraySize(g_fx); i++)     g_engine.ReleaseSymbol(g_fx[i]);
   for(int i = 0; i < ArraySize(g_crypto); i++) g_engine.ReleaseSymbol(g_crypto[i]);
   Comment("");
}

//+------------------------------------------------------------------+
bool IsNewBar(void)
{
   const datetime t = iTime(_Symbol, InpEntryTF, 0);
   if(t == g_last_bar_time) return false;
   g_last_bar_time = t;
   return true;
}

//+------------------------------------------------------------------+
bool IsWeekend(void)
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return (dt.day_of_week == 0 || dt.day_of_week == 6);
}

//+------------------------------------------------------------------+
void ManageOpenPositions(void)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      const long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic != InpMagicMeanRev && magic != InpMagicTrend && magic != InpMagicCrypto)
         continue;

      const string symbol   = PositionGetString(POSITION_SYMBOL);
      const long   type     = PositionGetInteger(POSITION_TYPE);
      const double open     = PositionGetDouble(POSITION_PRICE_OPEN);
      const double sl       = PositionGetDouble(POSITION_SL);
      const double volume   = PositionGetDouble(POSITION_VOLUME);
      const datetime opened = (datetime)PositionGetInteger(POSITION_TIME);

      if(TimeCurrent() - opened < InpMinHoldSeconds) continue;

      const double price = (type == POSITION_TYPE_BUY)
                           ? SymbolInfoDouble(symbol, SYMBOL_BID)
                           : SymbolInfoDouble(symbol, SYMBOL_ASK);
      if(price <= 0.0) continue;

      const string comment = PositionGetString(POSITION_COMMENT);
      const double r_unit  = ParseRUnit(comment);
      if(r_unit <= 0.0) continue;

      const double move = (type == POSITION_TYPE_BUY) ? (price - open) : (open - price);
      const double r_now = move / r_unit;

      // Partial profit realization
      if(r_now >= InpPartialCloseR && !IsPartialDone(comment))
      {
         const double close_vol = g_risk.NormalizeLots(symbol, volume * (InpPartialClosePct / 100.0));
         if(close_vol > 0.0 && close_vol < volume)
         {
            if(g_trade.PositionClosePartial(ticket, close_vol))
               PrintFormat("[MANAGE] Partial close executed: %.2f lots on %s at %.2fR.",
                           close_vol, symbol, r_now);
         }
      }

      // ATR-based trailing stop
      if(r_now >= InpTrailStartR)
      {
         const double atr_val = ATRForSymbol(symbol);
         if(atr_val > 0.0)
         {
            const double dist = atr_val * InpTrailAtrMult;
            double new_sl = (type == POSITION_TYPE_BUY) ? (price - dist) : (price + dist);
            new_sl = NormalizeDouble(new_sl, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS));

            const bool improves = (type == POSITION_TYPE_BUY)
                                  ? (sl == 0.0 || new_sl > sl)
                                  : (sl == 0.0 || new_sl < sl);

            if(improves && RespectsStopLevel(symbol, price, new_sl))
               g_trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
         }
      }
   }
}

//+------------------------------------------------------------------+
double ParseRUnit(const string comment)
{
   const int pos = StringFind(comment, "|");
   if(pos < 0) return 0.0;
   return StringToDouble(StringSubstr(comment, pos + 1));
}

//+------------------------------------------------------------------+
bool IsPartialDone(const string comment)
{
   return (StringFind(comment, "#P") >= 0);
}

//+------------------------------------------------------------------+
double ATRForSymbol(const string symbol)
{
   for(int i = 0; i < ArraySize(g_fx); i++)
      if(g_fx[i].symbol == symbol) return g_engine.ATR(g_fx[i]);
   for(int i = 0; i < ArraySize(g_crypto); i++)
      if(g_crypto[i].symbol == symbol) return g_engine.ATR(g_crypto[i]);
   return 0.0;
}

//+------------------------------------------------------------------+
bool RespectsStopLevel(const string symbol, const double price, const double sl)
{
   const long   level_pts = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   const double point     = SymbolInfoDouble(symbol, SYMBOL_POINT);
   const double min_dist  = level_pts * point;
   return MathAbs(price - sl) > min_dist;
}

//+------------------------------------------------------------------+
bool HasPositionOn(const string symbol, const long magic)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) == symbol &&
         PositionGetInteger(POSITION_MAGIC) == magic)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
void TryEnter(SymbolContext &ctx, const ENUM_SIGNAL sig, const long magic,
              const double risk_pct, const double sl_atr_mult, const bool use_tp_multiple)
{
   if(sig == SIGNAL_NONE) return;
   if(HasPositionOn(ctx.symbol, magic)) return;

   // Spread check filter
   const long spread = SymbolInfoInteger(ctx.symbol, SYMBOL_SPREAD);
   if(spread > InpMaxSpreadPoints)
   {
      PrintFormat("[ENTRY] Spread filter triggered on %s: %d > %d pts. Entry skipped.",
                  ctx.symbol, spread, InpMaxSpreadPoints);
      return;
   }

   const double atr_val = g_engine.ATR(ctx);
   if(atr_val <= 0.0) return;

   const double stop_dist = atr_val * sl_atr_mult;
   const double lots = g_risk.CalcLots(ctx.symbol, risk_pct, stop_dist, InpMaxRoomFraction);
   if(lots <= 0.0) return;

   const int digits = (int)SymbolInfoInteger(ctx.symbol, SYMBOL_DIGITS);
   const double ask = SymbolInfoDouble(ctx.symbol, SYMBOL_ASK);
   const double bid = SymbolInfoDouble(ctx.symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0) return;

   double entry, sl, tp;

   if(sig == SIGNAL_BUY)
   {
      entry = ask;
      sl    = NormalizeDouble(entry - stop_dist, digits);
      tp    = use_tp_multiple
              ? NormalizeDouble(entry + stop_dist * InpTrendTPRMultiple, digits)
              : NormalizeDouble(g_engine.BBMiddle(ctx), digits);
      if(tp <= entry) return;
   }
   else
   {
      entry = bid;
      sl    = NormalizeDouble(entry + stop_dist, digits);
      tp    = use_tp_multiple
              ? NormalizeDouble(entry - stop_dist * InpTrendTPRMultiple, digits)
              : NormalizeDouble(g_engine.BBMiddle(ctx), digits);
      if(tp >= entry) return;
   }

   if(!RespectsStopLevel(ctx.symbol, entry, sl)) return;

   const string comment = StringFormat("%s|%.*f",
                                       (magic == InpMagicTrend ? "TR" : "MR"),
                                       digits, stop_dist);

   g_trade.SetExpertMagicNumber(magic);
   g_trade.SetTypeFillingBySymbol(ctx.symbol);

   const bool ok = (sig == SIGNAL_BUY)
                   ? g_trade.Buy (lots, ctx.symbol, entry, sl, tp, comment)
                   : g_trade.Sell(lots, ctx.symbol, entry, sl, tp, comment);

   if(ok)
      PrintFormat("[ENTRY] %s executed: %s %.2f lots @ %.*f SL %.*f TP %.*f (risk: %.2f%%)",
                  (sig == SIGNAL_BUY ? "BUY" : "SELL"), ctx.symbol, lots,
                  digits, entry, digits, sl, digits, tp, risk_pct);
   else
      PrintFormat("[ENTRY] Order submission failed on %s: %d %s",
                  ctx.symbol, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
void OnTick()
{
   g_risk.Update();
   ManageOpenPositions();

   Comment("HybridQuantEA\n", g_risk.StatusLine());

   if(!g_risk.TradingAllowed()) return;
   if(!IsNewBar()) return;

   const int open_total = g_risk.OpenPositions(InpMagicMeanRev)
                        + g_risk.OpenPositions(InpMagicTrend)
                        + g_risk.OpenPositions(InpMagicCrypto);
   if(open_total >= InpMaxOpenPositions) return;

   // Weekday symbols
   if(!IsWeekend())
   {
      for(int i = 0; i < ArraySize(g_fx); i++)
      {
         if(!g_fx[i].ready) continue;

         TryEnter(g_fx[i], g_engine.MeanReversionSignal(g_fx[i]),
                  InpMagicMeanRev, InpRiskMeanRev, InpSLAtrMult, false);

         TryEnter(g_fx[i], g_engine.TrendSignal(g_fx[i], InpBreakoutLookback),
                  InpMagicTrend, InpRiskTrend, InpSLAtrMult, true);
      }
   }

   // 24/7 Crypto symbols
   if(InpTradeCrypto)
   {
      for(int i = 0; i < ArraySize(g_crypto); i++)
      {
         if(!g_crypto[i].ready) continue;

         ENUM_SIGNAL sig = g_engine.MeanReversionSignal(g_crypto[i]);
         if(sig == SIGNAL_NONE)
            sig = g_engine.TrendSignal(g_crypto[i], InpBreakoutLookback);

         TryEnter(g_crypto[i], sig, InpMagicCrypto,
                  InpRiskCrypto, InpSLAtrMultCrypto, true);
      }
   }
}
//+------------------------------------------------------------------+
