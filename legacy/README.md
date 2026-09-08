# Frozen legacy program

`mark_distortion_correct_se2_v1.py` is a byte-for-byte snapshot of the
laboratory's original SE2 correction script at the start of package migration.
It is retained for historical reproducibility and must not be edited.

- Frozen repository commit: `820c689479ce0f9f67bf1183eea17127fd42427f`
- Local recovery tag: `semcorr-prepackage-v0.1.0`
- SHA-256: `2271194df6b17e973f2477ed897c1737680dc4dc8256f947d2a503178e24e66a`

Run it directly only when reproducing an old analysis:

```bash
python legacy/mark_distortion_correct_se2_v1.py IMAGE.tif
```

New laboratory work should use the installed `semcorr` command.
