//+------------------------------------------------------------------+
//|                                               RiskManager.mqh     |
//|  Equity-based trailing drawdown protection and volatility sizing  |
//+------------------------------------------------------------------+
#property copyright "HybridQuant"
#property strict

//+------------------------------------------------------------------+
//| Risk Manager Class                                               |
//| Enforces ratcheting equity floor constraints, circuit breaker     |
//| logic, and double-capped position sizing.                        |
//+------------------------------------------------------------------+
class CRiskManager
{
private:
   double   m_initial_balance;
   double   m_dd_percent;
   double   m_equity_peak;
   double   m_soft_stop_ratio;
   bool     m_trading_halted;
   datetime m_halt_time;

public:
   CRiskManager(void) : m_initial_balance(0.0),
                        m_dd_percent(6.0),
                        m_equity_peak(0.0),
                        m_soft_stop_ratio(0.60),
                        m_trading_halted(false),
                        m_halt_time(0) {}

   void Init(const double initial_balance,
             const double dd_percent,
             const double soft_stop_ratio)
   {
      m_initial_balance = initial_balance;
      m_dd_percent      = dd_percent;
      m_soft_stop_ratio = soft_stop_ratio;
      m_equity_peak     = AccountInfoDouble(ACCOUNT_EQUITY);
      m_trading_halted  = false;
      m_halt_time       = 0;
   }

   void Update(void)
   {
      const double equity = AccountInfoDouble(ACCOUNT_EQUITY);

      if(equity > m_equity_peak)
         m_equity_peak = equity;

      if(!m_trading_halted && UsedDrawdownRatio() >= m_soft_stop_ratio)
      {
         m_trading_halted = true;
         m_halt_time      = TimeCurrent();
         PrintFormat("[RISK] Circuit breaker triggered. Equity=%.2f Floor=%.2f "
                     "Used=%.1f%% of DD budget. Disabling new order entries.",
                     equity, BreachFloor(), UsedDrawdownRatio() * 100.0);
      }
   }

   double BreachFloor(void) const
   {
      return m_equity_peak - (m_initial_balance * m_dd_percent / 100.0);
   }

   double RoomToFloor(void) const
   {
      return AccountInfoDouble(ACCOUNT_EQUITY) - BreachFloor();
   }

   double UsedDrawdownRatio(void) const
   {
      const double budget = m_initial_balance * m_dd_percent / 100.0;
      if(budget <= 0.0) return 1.0;
      return MathMax(0.0, MathMin(1.0, 1.0 - (RoomToFloor() / budget)));
   }

   bool TradingAllowed(void) const
   {
      return !m_trading_halted;
   }

   void ResetHalt(void)
   {
      m_trading_halted = false;
      m_halt_time      = 0;
   }

   double CalcLots(const string symbol,
                   const double risk_pct,
                   const double stop_distance_price,
                   const double max_room_fraction = 0.25)
   {
      if(stop_distance_price <= 0.0) return 0.0;

      const double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
      const double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
      if(tick_value <= 0.0 || tick_size <= 0.0) return 0.0;

      const double loss_per_lot = (stop_distance_price / tick_size) * tick_value;
      if(loss_per_lot <= 0.0) return 0.0;

      const double equity = AccountInfoDouble(ACCOUNT_EQUITY);

      // Cap 1: Conventional percent-of-equity risk
      const double risk_cap = equity * (risk_pct / 100.0);

      // Cap 2: Remaining room fraction to ratcheting floor
      const double room_cap = MathMax(0.0, RoomToFloor()) * max_room_fraction;

      const double money_at_risk = MathMin(risk_cap, room_cap);
      if(money_at_risk <= 0.0) return 0.0;

      const double raw_lots = money_at_risk / loss_per_lot;
      return NormalizeLots(symbol, raw_lots);
   }

   double NormalizeLots(const string symbol, double lots) const
   {
      const double min_lot   = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      const double max_lot   = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
      const double lot_step  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
      const double lot_limit = SymbolInfoDouble(symbol, SYMBOL_VOLUME_LIMIT);

      if(lot_step <= 0.0) return 0.0;

      lots = MathFloor(lots / lot_step) * lot_step;
      lots = MathMin(lots, max_lot);

      if(lot_limit > 0.0)
         lots = MathMin(lots, lot_limit);

      // Skip trade if calculated volume is below broker minimum
      if(lots < min_lot) return 0.0;

      const int digits = (int)MathMax(0, -MathLog10(lot_step) + 0.5);
      return NormalizeDouble(lots, digits);
   }

   double FloatingPnL(void) const
   {
      double total = 0.0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         const ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         total += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      }
      return total;
   }

   int OpenPositions(const long magic) const
   {
      int count = 0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         const ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if(PositionGetInteger(POSITION_MAGIC) == magic) count++;
      }
      return count;
   }

   string StatusLine(void) const
   {
      return StringFormat("Equity %.2f | Peak %.2f | Floor %.2f | Room %.2f | DD Used %.1f%% | %s",
                          AccountInfoDouble(ACCOUNT_EQUITY),
                          m_equity_peak,
                          BreachFloor(),
                          RoomToFloor(),
                          UsedDrawdownRatio() * 100.0,
                          m_trading_halted ? "HALTED" : "ACTIVE");
   }
};
