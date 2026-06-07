#!/usr/bin/env python3
"""
Script de verificación rápida del microservicio.
Ejecutar ANTES de hacer deploy:
    python scripts/test_local.py
"""
import sys
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"

def call(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ✗ HTTP {e.code}: {body[:200]}")
        return {}
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {}

def check(label: str, condition: bool, detail: str = ""):
    icon = "✓" if condition else "✗"
    print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))
    return condition

def main():
    print("\n══════════════════════════════════════")
    print("  StockLens Python — Test rápido")
    print("══════════════════════════════════════\n")

    failures = 0

    # 1. Health
    print("▸ Health check")
    h = call("/health")
    if not check("Status OK", h.get("status") == "ok"): failures += 1

    # 2. Technical
    print("\n▸ Análisis técnico (AAPL, 1y)")
    t = call("/analyze/technical/AAPL?period=1y")
    if not check("Tiene datos",   len(t.get("data", [])) > 100,  f"{len(t.get('data',[]))} sesiones"): failures += 1
    if not check("RSI calculado", t.get("data", [{}])[-1].get("rsi") is not None): failures += 1
    if not check("MACD calculado",t.get("data", [{}])[-1].get("macd") is not None): failures += 1
    if not check("Señales",       "signals" in t): failures += 1
    if "signals" in t:
        check("Trend detectada", t["signals"].get("trend") in ("bullish", "bearish", "neutral"),
              t["signals"].get("trend", "N/A"))

    # 3. Fundamental
    print("\n▸ Análisis fundamental (MSFT)")
    f = call("/analyze/fundamental/MSFT")
    if not check("Nombre empresa",   bool(f.get("company_name"))): failures += 1
    if not check("Métricas",         bool(f.get("metrics"))): failures += 1
    if not check("Value Score",      f.get("value_score", {}).get("score") is not None,
                 f"score={f.get('value_score',{}).get('score','?')}"): failures += 1
    if not check("Criterios Value",  len(f.get("value_score", {}).get("criteria", [])) >= 4): failures += 1

    # 4. Ticker inválido
    print("\n▸ Manejo de error (ticker inválido)")
    try:
        with urllib.request.urlopen(f"{BASE}/analyze/technical/XXXXXXXXXXX?period=1y", timeout=15) as r:
            check("Debería fallar", False, "Devolvió 200 OK en vez de 404")
            failures += 1
    except urllib.error.HTTPError as e:
        check("Error manejado correctamente", e.code in (404, 500), f"HTTP {e.code}")

    # Resumen
    print(f"\n══════════════════════════════════════")
    if failures == 0:
        print("  ✅ Todo OK — listo para deploy")
    else:
        print(f"  ❌ {failures} test(s) fallaron")
        sys.exit(1)
    print("══════════════════════════════════════\n")

if __name__ == "__main__":
    main()
