//+------------------------------------------------------------------+
//|                                                Strategies.mqh     |
//|  Signal generation: mean reversion, trend following, regime gate  |
//+------------------------------------------------------------------+
#property copyright "HybridQuant"
#property strict

enum ENUM_SIGNAL
{
   SIGNAL_NONE = 0,
   SIGNAL_BUY  = 1,
   SIGNAL_SELL = -1
};

enum ENUM_REGIME
{
   REGIME_UNKNOWN  = 0,
   REGIME_RANGING  = 1,
   REGIME_TRENDING = 2
};

struct SymbolContext
{
   string symbol;
   int    h_bb;
   int    h_rsi;
   int    h_atr;
   int    h_adx_htf;
   int    h_ma_fast;
   int    h_ma_slow;
   bool   ready;
};

class CStrategyEngine
{
private:
   ENUM_TIMEFRAMES m_tf_entry;
   ENUM_TIMEFRAMES m_tf_regime;

   int    m_bb_period;
   double m_bb_dev;
   int    m_rsi_period;
   double m_rsi_os;
   double m_rsi_ob;
   int    m_atr_period;
   int    m_adx_period;
   double m_adx_range_max;
   double m_adx_trend_min;
   int    m_ma_fast;
   int    m_ma_slow;

public:
   CStrategyEngine(void) : m_tf_entry(PERIOD_H1),
                           m_tf_regime(PERIOD_H4),
                           m_bb_period(20),
                           m_bb_dev(2.0),
                           m_rsi_period(14),
                           m_rsi_os(30.0),
                           m_rsi_ob(70.0),
                           m_atr_period(14),
                           m_adx_period(14),
                           m_adx_range_max(20.0),
                           m_adx_trend_min(25.0),
                           m_ma_fast(20),
                           m_ma_slow(50) {}

   void Configure(const ENUM_TIMEFRAMES tf_entry,
                  const ENUM_TIMEFRAMES tf_regime,
                  const int bb_period, const double bb_dev,
                  const int rsi_period, const double rsi_os, const double rsi_ob,
                  const int atr_period,
                  const int adx_period, const double adx_range_max,
                  const double adx_trend_min,
                  const int ma_fast, const int ma_slow)
   {
      m_tf_entry      = tf_entry;
      m_tf_regime     = tf_regime;
      m_bb_period     = bb_period;
      m_bb_dev        = bb_dev;
      m_rsi_period    = rsi_period;
      m_rsi_os        = rsi_os;
      m_rsi_ob        = rsi_ob;
      m_atr_period    = atr_period;
      m_adx_period    = adx_period;
      m_adx_range_max = adx_range_max;
      m_adx_trend_min = adx_trend_min;
      m_ma_fast       = ma_fast;
      m_ma_slow       = ma_slow;
   }

   bool InitSymbol(SymbolContext &ctx, const string symbol)
   {
      ctx.symbol = symbol;
      ctx.ready  = false;

      if(!SymbolSelect(symbol, true))
      {
         PrintFormat("[STRAT] Symbol %s unavailable in Market Watch.", symbol);
         return false;
      }

      ctx.h_bb      = iBands(symbol, m_tf_entry, m_bb_period, 0, m_bb_dev, PRICE_CLOSE);
      ctx.h_rsi     = iRSI(symbol, m_tf_entry, m_rsi_period, PRICE_CLOSE);
      ctx.h_atr     = iATR(symbol, m_tf_entry, m_atr_period);
      ctx.h_adx_htf = iADX(symbol, m_tf_regime, m_adx_period);
      ctx.h_ma_fast = iMA(symbol, m_tf_regime, m_ma_fast, 0, MODE_EMA, PRICE_CLOSE);
      ctx.h_ma_slow = iMA(symbol, m_tf_regime, m_ma_slow, 0, MODE_EMA, PRICE_CLOSE);

      if(ctx.h_bb      == INVALID_HANDLE || ctx.h_rsi     == INVALID_HANDLE ||
         ctx.h_atr     == INVALID_HANDLE || ctx.h_adx_htf == INVALID_HANDLE ||
         ctx.h_ma_fast == INVALID_HANDLE || ctx.h_ma_slow == INVALID_HANDLE)
      {
         PrintFormat("[STRAT] Failed to create indicator handles for %s.", symbol);
         return false;
      }

      ctx.ready = true;
      return true;
   }

   void ReleaseSymbol(SymbolContext &ctx)
   {
      if(ctx.h_bb      != INVALID_HANDLE) { IndicatorRelease(ctx.h_bb);      ctx.h_bb      = INVALID_HANDLE; }
      if(ctx.h_rsi     != INVALID_HANDLE) { IndicatorRelease(ctx.h_rsi);     ctx.h_rsi     = INVALID_HANDLE; }
      if(ctx.h_atr     != INVALID_HANDLE) { IndicatorRelease(ctx.h_atr);     ctx.h_atr     = INVALID_HANDLE; }
      if(ctx.h_adx_htf != INVALID_HANDLE) { IndicatorRelease(ctx.h_adx_htf); ctx.h_adx_htf = INVALID_HANDLE; }
      if(ctx.h_ma_fast != INVALID_HANDLE) { IndicatorRelease(ctx.h_ma_fast); ctx.h_ma_fast = INVALID_HANDLE; }
      if(ctx.h_ma_slow != INVALID_HANDLE) { IndicatorRelease(ctx.h_ma_slow); ctx.h_ma_slow = INVALID_HANDLE; }
      ctx.ready = false;
   }

   bool ReadBuffer(const int handle, const int buffer, const int shift, double &out) const
   {
      if(handle == INVALID_HANDLE) return false;
      if(BarsCalculated(handle) <= shift) return false;

      double tmp[];
      ArraySetAsSeries(tmp, true);
      if(CopyBuffer(handle, buffer, shift, 1, tmp) != 1) return false;
      out = tmp[0];
      return true;
   }

   ENUM_REGIME Regime(const SymbolContext &ctx) const
   {
      double adx_val;
      if(!ReadBuffer(ctx.h_adx_htf, 0, 1, adx_val)) return REGIME_UNKNOWN;

      if(adx_val <= m_adx_range_max) return REGIME_RANGING;
      if(adx_val >= m_adx_trend_min) return REGIME_TRENDING;
      return REGIME_UNKNOWN;
   }

   ENUM_SIGNAL MeanReversionSignal(const SymbolContext &ctx) const
   {
      if(Regime(ctx) != REGIME_RANGING) return SIGNAL_NONE;

      double upper, lower, rsi_val;
      if(!ReadBuffer(ctx.h_bb, 1, 1, upper))   return SIGNAL_NONE;
      if(!ReadBuffer(ctx.h_bb, 2, 1, lower))   return SIGNAL_NONE;
      if(!ReadBuffer(ctx.h_rsi, 0, 1, rsi_val)) return SIGNAL_NONE;

      const double close = iClose(ctx.symbol, m_tf_entry, 1);
      if(close <= 0.0) return SIGNAL_NONE;

      if(close < lower && rsi_val < m_rsi_os) return SIGNAL_BUY;
      if(close > upper && rsi_val > m_rsi_ob) return SIGNAL_SELL;
      return SIGNAL_NONE;
   }

   ENUM_SIGNAL TrendSignal(const SymbolContext &ctx, const int breakout_lookback = 20) const
   {
      if(Regime(ctx) != REGIME_TRENDING) return SIGNAL_NONE;

      double ma_fast, ma_slow;
      if(!ReadBuffer(ctx.h_ma_fast, 0, 1, ma_fast)) return SIGNAL_NONE;
      if(!ReadBuffer(ctx.h_ma_slow, 0, 1, ma_slow)) return SIGNAL_NONE;

      const double close = iClose(ctx.symbol, m_tf_entry, 1);
      if(close <= 0.0) return SIGNAL_NONE;

      const int hi_idx = iHighest(ctx.symbol, m_tf_entry, MODE_HIGH, breakout_lookback, 2);
      const int lo_idx = iLowest (ctx.symbol, m_tf_entry, MODE_LOW,  breakout_lookback, 2);
      if(hi_idx < 0 || lo_idx < 0) return SIGNAL_NONE;

      const double prior_high = iHigh(ctx.symbol, m_tf_entry, hi_idx);
      const double prior_low  = iLow (ctx.symbol, m_tf_entry, lo_idx);

      if(ma_fast > ma_slow && close > prior_high) return SIGNAL_BUY;
      if(ma_fast < ma_slow && close < prior_low)  return SIGNAL_SELL;
      return SIGNAL_NONE;
   }

   double ATR(const SymbolContext &ctx) const
   {
      double atr_val;
      if(!ReadBuffer(ctx.h_atr, 0, 1, atr_val)) return 0.0;
      return atr_val;
   }

   double BBMiddle(const SymbolContext &ctx) const
   {
      double mid;
      if(!ReadBuffer(ctx.h_bb, 0, 1, mid)) return 0.0;
      return mid;
   }

   ENUM_TIMEFRAMES EntryTimeframe(void) const { return m_tf_entry; }
};
