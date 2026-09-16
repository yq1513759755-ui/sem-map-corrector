#!/usr/bin/env python3
"""Generate AutoCAD placement from explicit bottom-left filename coordinates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcorr.cad import main
if __name__=='__main__':
    raise SystemExit(main())
