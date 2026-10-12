//+------------------------------------------------------------------+
//|                                                     xtbridge.mq4 |
//|  File-based MetaTrader5-style bridge for mt5_bot_v2.py           |
//|                                                                  |
//|  protocol (all in MQL4\Files\):                                  |
//|    python writes:  xtb_req_<pid>_<seq>.txt   (SEQ line + cmds)   |
//|    EA answers:     xtb_r<pid>_<seq>.txt      (SEQ line + results) |
//|    heartbeat:      xtb_alive.txt                               |
//|  EA processes every request file it finds, writes the response,   |
//|  then deletes the request file. No shared handles, no renames      |
//|  on top of each other - each side owns its files.                  |
//+------------------------------------------------------------------+
#property strict

#define REQ_PREFIX  "xtb_req_"
#define RESP_PREFIX "xtb_r"
#define ALIVE_FILE  "xtb_alive.txt"

int OnInit()
  {
   PurgeFiles(REQ_PREFIX "*");
   PurgeFiles(RESP_PREFIX "*.txt");
   if(!EventSetMillisecondTimer(150))
     {
      Print("xtbridge: cannot start timer");
      return(INIT_FAILED);
     }
   WriteAlive();
   Print("xtbridge: started - file bridge for python");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

void OnTick(void)
  {
   Pump();
  }

void OnTimer(void)
  {
   Pump();
   WriteAlive();
  }

//+------------------------------------------------------------------+
//| helpers                                                          |
//+------------------------------------------------------------------+
string FP(const double v, const int dig)
  {
   return (DoubleToString(NormalizeDouble(v, dig), dig));
  }

string F8(const double v)
  {
   return (DoubleToString(v, 8));
  }

string I(const long v)
  {
   return (IntegerToString(v));
  }

void Add(string &arr[], const string s)
  {
   int n = ArraySize(arr);
   ArrayResize(arr, n + 1);
   arr[n] = s;
  }

string ReadAll(const string name)
  {
   int h = FileOpen(name, FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
      return ("");
   string acc = "";
   while(!FileIsEnding(h))
      acc += FileReadString(h) + "\n";
   FileClose(h);
      return (acc);
  }

bool WriteAll(const string name, const string body)
  {
   int h = FileOpen(name, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
      return (false);
   FileWriteString(h, body);
   FileClose(h);
   return (true);
  }

void PurgeFiles(const string pattern)
  {
   string fname = "";
   int h = FileFindFirst(pattern, fname, 0);
   if(h == INVALID_HANDLE)
      return;
   string acc[];
   int n = 0;
   do
     {
      ArrayResize(acc, n + 1);
      acc[n] = fname;
      n++;
     }
   while(FileFindNext(h, fname));
   FileFindClose(h);
   for(int i = 0; i < n; i++)
      FileDelete(acc[i], 0);
  }

void WriteAlive(void)
  {
   string s = "alive|" + I(TimeLocal()) + "|" + I(AccountNumber()) + "|"
              + AccountServer() + "|"
              + I(TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ? 1 : 0) + "|"
              + I(TerminalInfoInteger(TERMINAL_CONNECTED) ? 1 : 0);
   // write to a private tmp then move into place (best effort)
   if(WriteAll("xtb_alive.tmp", s))
      FileMove("xtb_alive.tmp", 0, ALIVE_FILE, FILE_REWRITE);
  }

//+------------------------------------------------------------------+
//| error mapping: MT4 GetLastError -> MT5 TRADE_RETCODE            |
//+------------------------------------------------------------------+
string Ret(const int e)
  {
   switch(e)
     {
      case 0:   return ("10009");
      case 1:   return ("10009");   // no changes = desired state reached
      case 3:   return ("10015");   // invalid price
      case 129: return ("10015");
      case 130: return ("10016");   // invalid stops
      case 131: return ("10013");   // invalid trade parameters
      case 132: return ("10018");   // market closed
      case 133: return ("10015");   // invalid price
      case 134: return ("10019");   // not enough money
      case 135: return ("10010");   // price changed / requote
      case 136: return ("10011");   // off quotes
      case 137: return ("10010");   // broker busy
      case 138: return ("10010");   // requote
      case 141: return ("10041");   // too many requests
      case 146: return ("10021");   // trade context busy
      case 147: return ("10022");   // trade requests denied
      case 148: return ("10018");   // deal timeout / open limit
      case 149: return ("10015");   // hedge forbidden
      case 150: return ("10021");   // tp/blocked by FIFO etc
      case 151: return ("10013");   // tp invalid
      case 152: return ("10013");   // sl invalid
      case 161: return ("10004");   // trade disabled
      case 167: return ("10004");   // trade disabled by symbol
      case 169: return ("10015");   // quotes
      case 170: return ("10016");   // permanent restriction
      default:  return ("10008");   // reject
     }
  }

string ErrStr(const int e)
  {
   switch(e)
     {
      case 0:   return ("ok");
      case 1:   return ("no changes");
      case 130: return ("invalid stops");
      case 131: return ("invalid params");
      case 132: return ("market closed");
      case 133: return ("invalid price");
      case 134: return ("not enough money");
      case 135: return ("price changed");
      case 136: return ("no quotes");
      case 137: return ("broker busy");
      case 138: return ("requote");
      case 146: return ("trade context busy");
      case 148: return ("timeout");
      case 161: return ("trade disabled");
      case 164: return ("too many requests");
      case 167: return ("market closed for symbol");
      case 169: return ("no quotes");
      default:  return ("error " + I(e));
     }
  }

double NormLot(const string sym, const double lot, bool &ok)
  {
   double step = MarketInfo(sym, MODE_LOTSTEP);
   double mn = MarketInfo(sym, MODE_MINLOT);
   double mx = MarketInfo(sym, MODE_MAXLOT);
   if(step <= 0)
      step = 0.01;
   double l = MathFloor(lot / step + 1e-9) * step;
   ok = true;
   if(mn > 0 && l < mn - 1e-12)
     {
      ok = false;
      return (0);
     }
   if(mx > 0 && l > mx)
      l = mx;
   return (NormalizeDouble(l, 8));
  }

//+------------------------------------------------------------------+
//| request pump                                                     |
//+------------------------------------------------------------------+
void Pump(void)
  {
   // SINGLE-INSTANCE GUARD: serve the bridge from the M1 chart only.
   // Copies left on other timeframes re-execute every request (the
   // duplicate-ticket incident) - they stay silent.
   if(Period() != PERIOD_M1)
      return;
   string fname = "";
   int h = FileFindFirst(REQ_PREFIX "*.txt", fname, 0);
   if(h == INVALID_HANDLE)
      return;
   string names[];
   int nn = 0;
   do
     {
      ArrayResize(names, nn + 1);
      names[nn] = fname;
      nn++;
     }
   while(FileFindNext(h, fname));
   FileFindClose(h);
   for(int k = 0; k < nn; k++)
     {
      // never touch an in-flight .part write (8.3 wildcard can match it)
      if(StringFind(names[k], ".part") >= 0)
         continue;
      string body = ReadAll(names[k]);
      if(StringLen(body) == 0)
        {
         FileDelete(names[k], 0);   // stray/empty - clean up
         continue;
        }
      string lines[];
      int lc = StringSplit(body, (ushort)10, lines);
      string cmds[];
      int nc = 0;
      string seq_line = "";
      for(int li = 0; li < lc; li++)
        {
         string cmd = lines[li];
         int clen = StringLen(cmd);
         if(clen > 0 && StringGetCharacter(cmd, clen - 1) == 13)
            cmd = StringSubstr(cmd, 0, clen - 1);
         if(StringLen(cmd) == 0)
            continue;
         if(StringGetCharacter(cmd, 0) == 83
            && StringSubstr(cmd, 0, 4) == "SEQ|")
           {
            seq_line = cmd;
            continue;
           }
         ArrayResize(cmds, nc + 1);
         cmds[nc] = cmd;
         nc++;
        }
      // DUP-EXEC GUARD: own the request file BEFORE any side effect.
      // If the delete fails (AV lock etc.) we must NOT execute - the next
      // pump would re-find the same file and re-send the trade.
      if(!FileDelete(names[k], 0))
         continue;   // leave it; retry next pump (python has timeout+cleanup)
      string outs[];
      ArrayResize(outs, 0);
      for(int ci = 0; ci < nc; ci++)
         HandleCmd(cmds[ci], outs);
      string resp = seq_line + "\n";
      for(int oi = 0; oi < ArraySize(outs); oi++)
         resp += outs[oi] + "\n";
      string rname = RESP_PREFIX
                     + StringSubstr(names[k], StringLen(REQ_PREFIX));
      WriteAll(rname, resp);
     }
  }

//+------------------------------------------------------------------+
//| command dispatch                                                 |
//+------------------------------------------------------------------+
void HandleCmd(const string cmd, string &out[])
  {
   string p[];
   int pc = StringSplit(cmd, (ushort)124, p);   // '|'
   if(pc < 1)
      return;
   string c = p[0];
   if(c == "ACC")
      CmdACC(out);
   else if(c == "TI")
      CmdTI(out);
   else if(c == "SEL" && pc >= 2)
      Add(out, "SEL|" + I(MarketInfo(p[1], MODE_POINT) > 0 ? 1 : 0));
   else if(c == "SYM" && pc >= 2)
      CmdSYM(p[1], out);
   else if(c == "TICK" && pc >= 2)
      CmdTICK(p[1], out);
   else if(c == "BARS" && pc >= 4)
      CmdBARS(p[1], (int)StringToInteger(p[2]), (int)StringToInteger(p[3]), out);
   else if(c == "POS" && pc >= 2)
      CmdPOS(p[1], out);
   else if(c == "ORD" && pc >= 2)
      CmdORD(p[1], out);
   else if(c == "MAR" && pc >= 3)
      CmdMAR(p[1], StringToDouble(p[2]), out);
   else if(c == "SPO" && pc >= 10)
      CmdOpen(p, out);
   else if(c == "SPC" && pc >= 9)
      CmdClose(p, out);
   else if(c == "SPP" && pc >= 9)
      CmdPending(p, out);
   else if(c == "MOD" && pc >= 4)
      CmdMod(p, out);
   else if(c == "RMV" && pc >= 2)
      CmdRmv(p, out);
   else if(c == "HPH" && pc >= 2)
      CmdHph((long)StringToInteger(p[1]), out);
   else if(c == "HSYN" && pc >= 3)
      CmdHsyn((long)StringToInteger(p[1]), (long)StringToInteger(p[2]), out);
   else
      Add(out, "E|unknown:" + c);
  }

void CmdACC(string &out[])
  {
   int mode = (int)AccountInfoInteger(ACCOUNT_TRADE_MODE); // 0=demo
   Add(out, "ACC|" + I(AccountNumber()) + "|" + F8(AccountBalance())
       + "|" + F8(AccountEquity()) + "|" + F8(AccountFreeMargin())
       + "|" + F8(AccountMargin()) + "|" + F8(AccountProfit())
       + "|" + AccountServer() + "|" + I(mode)
       + "|" + I(AccountLeverage()) + "|" + AccountCurrency());
  }

void CmdTI(string &out[])
  {
   Add(out, "TI|"
       + I(TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ? 1 : 0)
       + "|" + I(MQLInfoInteger(MQL_TRADE_ALLOWED) ? 1 : 0)
       + "|" + I(TerminalInfoInteger(TERMINAL_CONNECTED) ? 1 : 0));
  }

void CmdSYM(const string sym, string &out[])
  {
   double pt = MarketInfo(sym, MODE_POINT);
   if(pt <= 0)
     {
      Add(out, "SYM|0");
      return;
     }
   int dig = (int)MarketInfo(sym, MODE_DIGITS);
   Add(out, "SYM|1|" + I(dig) + "|" + FP(pt, dig)
       + "|" + I((int)MarketInfo(sym, MODE_SPREAD))
       + "|" + F8(MarketInfo(sym, MODE_MINLOT))
       + "|" + F8(MarketInfo(sym, MODE_MAXLOT))
       + "|" + F8(MarketInfo(sym, MODE_LOTSTEP))
       + "|" + F8(MarketInfo(sym, MODE_LOTSIZE))
       + "|" + I((int)MarketInfo(sym, MODE_STOPLEVEL))
       + "|" + I((int)MarketInfo(sym, MODE_FREEZELEVEL))
       + "|" + F8(MarketInfo(sym, MODE_TICKVALUE))
       + "|" + F8(MarketInfo(sym, MODE_TICKSIZE))
       + "|" + F8(MarketInfo(sym, MODE_SWAPLONG))
       + "|" + F8(MarketInfo(sym, MODE_SWAPSHORT))
       + "|" + I((int)AccountInfoInteger(ACCOUNT_TRADE_MODE))
       + "|" + FP(MarketInfo(sym, MODE_BID), dig)
       + "|" + FP(MarketInfo(sym, MODE_ASK), dig)
       + "|2");   // filling bits: 2 = IOC
  }

void CmdTICK(const string sym, string &out[])
  {
   int dig = (int)MarketInfo(sym, MODE_DIGITS);
   double bid = MarketInfo(sym, MODE_BID);
   double ask = MarketInfo(sym, MODE_ASK);
   if(bid <= 0 || ask <= 0)
     {
      Add(out, "TICK|0");
      return;
     }
   Add(out, "TICK|1|" + FP(bid, dig) + "|" + FP(ask, dig)
       + "|" + I((long)MarketInfo(sym, MODE_TIME)) + "|0");
  }

void CmdBARS(const string sym, const int per, const int count, string &out[])
  {
   if(MarketInfo(sym, MODE_POINT) <= 0)
     {
      Add(out, "BARS|0");
      return;
     }
   int total = iBars(sym, per);
   if(total < 1)
     {
      Add(out, "BARS|0");
      return;
     }
   int take = count;
   if(take > total)
      take = total;
   if(take > 5000)
      take = 5000;
   if(take < 1)
     {
      Add(out, "BARS|0");
      return;
     }
   int dig = (int)MarketInfo(sym, MODE_DIGITS);
   Add(out, "BARS|1|" + I(take));
   for(int i = take - 1; i >= 0; i--)
     {
      Add(out, "B|" + I((long)iTime(sym, per, i))
          + "|" + FP(iOpen(sym, per, i), dig)
          + "|" + FP(iHigh(sym, per, i), dig)
          + "|" + FP(iLow(sym, per, i), dig)
          + "|" + FP(iClose(sym, per, i), dig)
          + "|0|0");
     }
  }

void CmdPOS(const string sym, string &out[])
  {
   bool any = (sym == "*");
   for(int i = 0; i < OrdersTotal(); i++)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderType() > OP_SELL)
         continue;
      if(!any && OrderSymbol() != sym)
         continue;
      Add(out, "P|" + I(OrderTicket())
          + "|" + I(OrderType())
          + "|" + FP(OrderOpenPrice(), (int)MarketInfo(OrderSymbol(), MODE_DIGITS))
          + "|" + F8(OrderLots())
          + "|" + F8(OrderProfit())
          + "|" + FP(OrderStopLoss(), (int)MarketInfo(OrderSymbol(), MODE_DIGITS))
          + "|" + FP(OrderTakeProfit(), (int)MarketInfo(OrderSymbol(), MODE_DIGITS))
          + "|" + I(OrderMagicNumber())
          + "|" + I((long)OrderOpenTime())
          + "|" + F8(OrderSwap())
          + "|" + OrderSymbol());
     }
  }

void CmdORD(const string sym, string &out[])
  {
   bool any = (sym == "*");
   for(int i = 0; i < OrdersTotal(); i++)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderType() <= OP_SELL)
         continue;                       // market positions -> POS
      if(!any && OrderSymbol() != sym)
         continue;
      int dig = (int)MarketInfo(OrderSymbol(), MODE_DIGITS);
      Add(out, "O|" + I(OrderTicket())
          + "|" + I(OrderType())
          + "|" + FP(OrderOpenPrice(), dig)
          + "|" + F8(OrderLots())
          + "|" + FP(OrderStopLoss(), dig)
          + "|" + FP(OrderTakeProfit(), dig)
          + "|" + I(OrderMagicNumber())
          + "|" + I((long)OrderOpenTime())
          + "|" + OrderSymbol());
     }
  }

void CmdMAR(const string sym, const double vol, string &out[])
  {
   double mr = MarketInfo(sym, MODE_MARGINREQUIRED);
   if(mr <= 0)
     {
      double lev = (double)AccountLeverage();
      double contract = MarketInfo(sym, MODE_LOTSIZE);
      double px = MarketInfo(sym, MODE_BID);
      if(lev > 0 && contract > 0 && px > 0)
         mr = vol * contract * px / lev;
      else
        {
         Add(out, "MAR|0");
         return;
        }
      Add(out, "MAR|" + F8(mr));
      return;
     }
   Add(out, "MAR|" + F8(mr * vol));
  }

//+------------------------------------------------------------------+
//| trade commands                                                   |
//+------------------------------------------------------------------+
void CmdOpen(string &p[], string &out[])
  {
   string sym = p[1];
   int type = (int)StringToInteger(p[2]);
   double vol = StringToDouble(p[3]);
   double sl = StringToDouble(p[5]);
   double tp = StringToDouble(p[6]);
   int slip = (int)StringToInteger(p[7]);
   long magic = StringToInteger(p[8]);
   string cmt = p[9];
   StringReplace(cmt, "|", "/");
   bool ok;
   double lot = NormLot(sym, vol, ok);
   if(!ok)
     {
      Add(out, "R|10014|0|0|volume " + F8(vol) + " below min/step");
      return;
     }
   int dig = (int)MarketInfo(sym, MODE_DIGITS);
   double px = (type == OP_BUY) ? MarketInfo(sym, MODE_ASK)
                                : MarketInfo(sym, MODE_BID);
   px = NormalizeDouble(px, dig);
   if(sl > 0)
      sl = NormalizeDouble(sl, dig);
   if(tp > 0)
      tp = NormalizeDouble(tp, dig);
   color clr = (type == OP_BUY) ? clrDodgerBlue : clrOrangeRed;
   ResetLastError();
   int ticket = OrderSend(sym, type, lot, px, slip, sl, tp, cmt,
                          (int)magic, 0, clr);
   if(ticket < 0)
     {
      int e = GetLastError();
      Add(out, "R|" + Ret(e) + "|0|0|MT4 " + ErrStr(e));
      return;
     }
   double done = px;
   if(OrderSelect(ticket, SELECT_BY_TICKET))
      done = OrderOpenPrice();
   Add(out, "R|10009|" + I(ticket) + "|" + FP(done, dig) + "|filled");
  }

void CmdClose(string &p[], string &out[])
  {
   long pos = StringToInteger(p[4]);
   double vol = StringToDouble(p[3]);
   int slip = (int)StringToInteger(p[6]);
   if(!OrderSelect((int)pos, SELECT_BY_TICKET) || OrderCloseTime() != 0)
     {
      Add(out, "R|10008|" + I(pos) + "|0|position not found or closed");
      return;
     }
   if(OrderType() > OP_SELL)
     {
      Add(out, "R|10008|" + I(pos) + "|0|ticket is pending, use RMV");
      return;
     }
   string sym = OrderSymbol();
   int dig = (int)MarketInfo(sym, MODE_DIGITS);
   double vol_avail = OrderLots();
   if(vol > vol_avail)
      vol = vol_avail;
   bool ok;
   double lot = NormLot(sym, vol, ok);
   if(!ok)
     {
      Add(out, "R|10014|" + I(pos) + "|0|volume " + F8(vol) + " invalid");
      return;
     }
   int cls = (OrderType() == OP_BUY) ? OP_SELL : OP_BUY;
   double px = (cls == OP_BUY) ? MarketInfo(sym, MODE_ASK)
                               : MarketInfo(sym, MODE_BID);
   px = NormalizeDouble(px, dig);
   color clr = (cls == OP_BUY) ? clrDodgerBlue : clrOrangeRed;
   ResetLastError();
   bool done = OrderClose((int)pos, lot, px, slip, clr);
   if(!done)
     {
      int e = GetLastError();
      Add(out, "R|" + Ret(e) + "|" + I(pos) + "|0|MT4 " + ErrStr(e));
      return;
     }
   Add(out, "R|10009|" + I(pos) + "|" + FP(px, dig) + "|closed");
  }

void CmdPending(string &p[], string &out[])
  {
   string sym = p[1];
   int type = (int)StringToInteger(p[2]);
   double vol = StringToDouble(p[3]);
   double price = StringToDouble(p[4]);
   double sl = StringToDouble(p[5]);
   double tp = StringToDouble(p[6]);
   long magic = StringToInteger(p[7]);
   string cmt = p[8];
   StringReplace(cmt, "|", "/");
   bool ok;
   double lot = NormLot(sym, vol, ok);
   if(!ok)
     {
      Add(out, "R|10014|0|0|volume " + F8(vol) + " below min/step");
      return;
     }
   int dig = (int)MarketInfo(sym, MODE_DIGITS);
   price = NormalizeDouble(price, dig);
   if(sl > 0)
      sl = NormalizeDouble(sl, dig);
   if(tp > 0)
      tp = NormalizeDouble(tp, dig);
   color clr = (type == OP_BUYLIMIT || type == OP_BUYSTOP)
               ? clrDodgerBlue : clrOrangeRed;
   ResetLastError();
   int ticket = OrderSend(sym, type, lot, price, 0, sl, tp, cmt,
                          (int)magic, 0, clr);
   if(ticket < 0)
     {
      int e = GetLastError();
      Add(out, "R|" + Ret(e) + "|0|0|MT4 " + ErrStr(e));
      return;
     }
   double done = price;
   if(OrderSelect(ticket, SELECT_BY_TICKET))
      done = OrderOpenPrice();
   Add(out, "R|10009|" + I(ticket) + "|" + FP(done, dig) + "|placed");
  }

void CmdMod(string &p[], string &out[])
  {
   long t = StringToInteger(p[1]);
   double nsl = StringToDouble(p[2]);
   double ntp = StringToDouble(p[3]);
   if(!OrderSelect((int)t, SELECT_BY_TICKET) || OrderCloseTime() != 0)
     {
      Add(out, "R|10008|" + I(t) + "|0|order not found or closed");
      return;
     }
   string sym = OrderSymbol();
   int dig = (int)MarketInfo(sym, MODE_DIGITS);
   double op = OrderOpenPrice();
   double sl = (nsl > 0) ? NormalizeDouble(nsl, dig) : 0.0;
   double tp = (ntp > 0) ? NormalizeDouble(ntp, dig) : 0.0;
   ResetLastError();
   bool done = OrderModify((int)t, op, sl, tp, 0, clrDodgerBlue);
   if(!done)
     {
      int e = GetLastError();
      if(e == 1)
        {
         Add(out, "R|10009|" + I(t) + "|" + FP(op, dig) + "|no changes");
         return;
        }
      Add(out, "R|" + Ret(e) + "|" + I(t) + "|0|MT4 " + ErrStr(e));
      return;
     }
   Add(out, "R|10009|" + I(t) + "|" + FP(op, dig) + "|modified");
  }

void CmdRmv(string &p[], string &out[])
  {
   long t = StringToInteger(p[1]);
   if(!OrderSelect((int)t, SELECT_BY_TICKET) || OrderCloseTime() != 0)
     {
      Add(out, "R|10008|" + I(t) + "|0|order not found or closed");
      return;
     }
   if(OrderType() <= OP_SELL)
     {
      Add(out, "R|10008|" + I(t) + "|0|position not removable, use SPC");
      return;
     }
   ResetLastError();
   bool done = OrderDelete((int)t, clrOrangeRed);
   if(!done)
     {
      int e = GetLastError();
      Add(out, "R|" + Ret(e) + "|" + I(t) + "|0|MT4 " + ErrStr(e));
      return;
     }
   Add(out, "R|10009|" + I(t) + "|0|deleted");
  }

//+------------------------------------------------------------------+
//| history                                                          |
//| H|ticket|type|opentime|closetime|openpx|closepx|vol|comm|swap|   |
//|   profit|sl|tp|magic|symbol|comment                              |
//+------------------------------------------------------------------+
string HRow(void)
  {
   string sym = OrderSymbol();
   int dig = (int)MarketInfo(sym, MODE_DIGITS);
   string cmt = OrderComment();
   StringReplace(cmt, "|", "/");
   return ("H|" + I(OrderTicket())
           + "|" + I(OrderType())
           + "|" + I((long)OrderOpenTime())
           + "|" + I((long)OrderCloseTime())
           + "|" + FP(OrderOpenPrice(), dig)
           + "|" + FP(OrderClosePrice(), dig)
           + "|" + F8(OrderLots())
           + "|" + F8(OrderCommission())
           + "|" + F8(OrderSwap())
           + "|" + F8(OrderProfit())
           + "|" + FP(OrderStopLoss(), dig)
           + "|" + FP(OrderTakeProfit(), dig)
           + "|" + I(OrderMagicNumber())
           + "|" + sym
           + "|" + cmt);
  }

void CmdHph(const long ticket, string &out[])
  {
   if(!OrderSelect((int)ticket, SELECT_BY_TICKET) || OrderCloseTime() == 0)
     {
      Add(out, "HPH|0");
      return;
     }
   Add(out, HRow());
  }

void CmdHsyn(const long from_ts, const long to_ts, string &out[])
  {
   int total = OrdersHistoryTotal();
   string rows[];
   ArrayResize(rows, 0);
   int kept = 0;
   // newest first, cap 1500, then re-emit oldest-first
   for(int i = total - 1; i >= 0; i--)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
         continue;
      if(OrderType() > OP_SELL)
         continue;
      long ct = (long)OrderCloseTime();
      if(ct < from_ts || ct > to_ts)
         continue;
      Add(rows, HRow());
      kept++;
      if(kept >= 1500)
         break;
     }
   if(kept == 0)
     {
      Add(out, "HSYN|0");
      return;
     }
   Add(out, "HSYN|" + I(kept));
   for(int j = kept - 1; j >= 0; j--)
      Add(out, rows[j]);
  }
//+------------------------------------------------------------------+
