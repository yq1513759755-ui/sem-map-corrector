#!/usr/bin/env python3
"""Generate AutoCAD placement from marker-row-column-sequence region filenames."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcorr.cad import main
if __name__=='__main__':
    raise SystemExit(main())
