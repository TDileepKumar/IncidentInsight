# Dataset Baseline Manifest — `7126c46`

This directory preserves a **historical dataset baseline** byte-for-byte. It is a
copy, not a regenerated dataset. Nothing here was produced by running
`data_generator.py`.

## 1. Baseline identifier and provenance

| Item | Value |
|---|---|
| Baseline identifier | `7126c46` |
| Source repository | IncidentInsight (local checkout path omitted — see `<REPO_ROOT>` in section 10) |
| Baseline HEAD (full) | `7126c4666ba590f08a9426222307026ab89f35e1` |
| Baseline HEAD subject | `fix: pluralize single-entry cluster labels` |
| Branch at capture | `main` |
| Working tree at capture | clean (`git status --short` empty) |
| Capture date | 2026-10-29 |

The identifier is the abbreviated HEAD commit at which the baseline was captured.
It records **when the copy was taken**, not when the data was originally generated.

### Original source paths (ignored, untracked)

| File | Original path | Original mtime |
|---|---|---|
| Raw log | `data/raw/synthetic_logs.log` | 2026-10-17 14:54:46 |
| Ground truth | `data/processed/ground_truth.csv` | 2026-10-17 14:54:46 |
| Metadata | `data/processed/generation_meta.json` | 2026-10-19 18:45:27 |

These originals are ignored by `.gitignore:17-19` and are **not** in Git history.
They were left untouched by the capture.

### Backup paths

| Copy | Location |
|---|---|
| Tracked copy (this directory) | `data/baseline/7126c46/` |
| External byte-exact backup | `<BACKUP_ROOT>/IncidentInsight-baseline-7126c46/` (maintainer's local copy, outside the repository) |

The external backup is outside the repository, so no Git newline filter applies to
it and its bytes match the originals exactly.

### How the copies were made

Byte-preserving file copy (`cp`), never a text-mode read-and-rewrite. All three
files were verified to hash identically to their originals immediately after
copying, in both destinations.

## 2. SHA-256 hashes

### Original byte-level (as-is on disk, CRLF)

These are the authoritative fingerprints of the baseline content. They apply
identically to the original files, the tracked copies **as they exist on disk
right now**, and the external backup files.

```
8c0e7392c5bcd9a3f0420c7335a817cdf13020409f0e42cc1d0b7d5aa64e307d  synthetic_logs.log
3d2af3a3d7816f6693d2057f2cf6a1a11b0164589d2f5391c70075efb1cad132  ground_truth.csv
cfe5bac583ad5aa52eeda440cd4412d7443359cff90099b37c17976e395bb5f4  generation_meta.json
```

### LF-normalized (the form Git will store — see section 3)

```
825ce10d993665b10d2136d527c78669784badd1030c1f99fa691d6b6961d81d  synthetic_logs.log
1f8d013f1d6c69e9313706de4d37b886a9a0a5654cc728dd635ae1df1c86a88d  ground_truth.csv
a078c51f10b4910d0b64ba16a36489aa97b80add5aee931dea8c81d03010da65  generation_meta.json
```

Each LF-normalized hash was computed over the original bytes with every `CRLF`
pair replaced by a single `LF`. No other transformation was applied.

## 3. CRLF-to-LF Git normalization caveat — read this before verifying hashes

The repository's `.gitattributes` contains:

```
* text=auto eol=lf
```

That policy was **kept unchanged**. No byte-preservation exception (such as
`-text`) was added for this directory.

**Consequence:** when these files are eventually committed, Git will normalize
their `CRLF` line endings to `LF` in the stored blobs. The committed blobs will
therefore **NOT retain the original CRLF bytes**, and their content will hash to
the **LF-normalized** values in section 2, not the original byte-level values.

This was confirmed empirically at capture time with `git hash-object` (run without
`-w`, so nothing was written). The blob ID computed **with** attribute filters
differs from the blob ID computed with `--no-filters` for all three files:

| File | Blob ID with filters (LF) | Blob ID `--no-filters` (CRLF) |
|---|---|---|
| `synthetic_logs.log` | `47cada53d1bc7bf2137054f8eda8804238fb394e` | `fd91b21d507a91625b0870d67982270a687505ba` |
| `ground_truth.csv` | `5ecab96fdd11ca55787a31fa1db64335e67a61f5` | `56c7538cd5872cb8389889608165cd5984f511b7` |
| `generation_meta.json` | `e51b7a50370190eddf1f1e02f36c0d95b3af5317` | `b6cc7373de7a23c467eb3748d3366208daa33074` |

Practical implications:

- A fresh `git clone` will check out **LF** versions of these files. Their SHA-256
  will match the LF-normalized values, and will **not** match the original
  byte-level values. This is expected, not corruption.
- The **external backup** is the only copy guaranteed to remain byte-exact to the
  original CRLF bytes, because it lives outside Git's control.
- Content is unaffected: the pipeline and tests read lines with `splitlines()`,
  which handles CRLF and LF identically.

## 4. File sizes and record counts

| File | Bytes (CRLF) | Bytes (LF) | Lines | Records |
|---|---|---|---|---|
| `synthetic_logs.log` | 107,002 | 106,002 | 1,000 | 1,000 log lines |
| `ground_truth.csv` | 109,418 | 108,417 | 1,001 | 1,000 rows + 1 header |
| `generation_meta.json` | 636 | 614 | 22 | 1 JSON object |

Combined original size: **217,056 bytes (212.0 KiB)**.

## 5. Ground-truth schema and category counts

**Columns (6), in file order:**

```
line_id, timestamp, level, category, request_id, message
```

There is **no `service` column**, although the raw log lines do contain a
`[service]` bracket. `line_id` is `int64`; all other columns are string dtype.
There are **zero null values** in any column. `line_id` is unique and equals
`0..999` in order.

**Category counts (actual baseline data):**

| Category | Count |
|---|---|
| `db_connection_failure` | 220 |
| `db_timeout` | 180 |
| `request_timeout` | 150 |
| `auth_failure` | 140 |
| `null_pointer` | 120 |
| `payment_gateway_error` | 90 |
| `file_not_found` | 60 |
| `validation_error` | 40 |
| **Total** | **1,000** |

## 6. Severity counts

| Level | Ground truth | Raw log |
|---|---|---|
| `ERROR` | 612 | 612 |
| `WARNING` | 244 | 244 |
| `CRITICAL` | 144 | 144 |
| `FATAL` | 0 | 0 |

The two distributions agree exactly. Note that **`CRITICAL` is present and `FATAL`
is absent** — see section 9.

Other observed characteristics: `request_id` is `req-` followed by **6** hex
characters in all 1,000 rows; timestamps span `2026-10-01T00:22:23Z` to
`2026-10-07T23:45:45Z`; there are **266 distinct messages**.

## 7. Raw-log / ground-truth alignment — verified

The raw log carries no explicit `line_id`; alignment is **positional** (raw line
*N* corresponds to the ground-truth row whose `line_id` is *N*).

All 1,000 rows were checked individually at capture time, and re-checked against
the copies afterwards. For each row, the corresponding raw line was required to:

1. start with `"{timestamp} {level} [{request_id}] "`, and
2. end with `" {message}"`.

**Result: 0 mismatches out of 1,000.** All 1,000 raw log lines align positionally
with the ground-truth rows.

Scope limitation: because the CSV has no `service` column, the `[service]` bracket
in each raw line could not be cross-checked against the answer key. Alignment is
confirmed for timestamp, level, request_id and message only.

## 8. Metadata contents — preserved verbatim, and NOT accurate

`generation_meta.json` was copied **verbatim, including its known
inconsistencies**. It was not corrected, regenerated, or rewritten.

| Field | Value |
|---|---|
| `seed` | 42 |
| `size` | 1000 |
| `generated_at` | `2026-10-14T16:35:09Z` |
| `log_start` | `2026-10-01T00:00:00Z` |
| `log_window_days` | 7 |

### WARNING: the metadata counts do not match the actual baseline data

Both count fields in the metadata contradict the data files it sits beside:

- `meta.category_counts` **does not equal** the ground-truth category counts in
  section 5. The metadata claims `request_timeout 180, db_timeout 150,
  auth_failure 130, validation_error 110, payment_gateway_error 60,
  file_not_found 30`, whereas the data actually contains `db_timeout 180,
  request_timeout 150, auth_failure 140, payment_gateway_error 90,
  file_not_found 60, validation_error 40`.
- `meta.level_counts` claims `{ERROR: 654, WARNING: 266, FATAL: 80}`. The actual
  data contains `{ERROR: 612, WARNING: 244, CRITICAL: 144}` and **no FATAL at all**.

The metadata's category counts instead match `config.CATEGORY_WEIGHTS`
(`src/config.py:45-54`) — that is, they describe what the **current** generator
would produce, not this baseline.

Timing evidence that the three files were not written by a single run: the raw log
and CSV both have mtime **2026-10-17 14:54:46**, while the metadata has mtime
**2026-10-19 18:45:27** and an embedded `generated_at` of **2026-10-14T16:35:09Z**
— three different times.

**Do not treat this metadata as a valid description of the baseline data.** It is
retained solely as historical provenance. The authoritative statistics are those
in sections 4-7, which were measured directly from the data files.

No claim is made here about how the mismatch arose; the cause was not established
from the available evidence.

## 9. WARNING: the current generator does NOT reproduce this baseline

Running `python data_generator.py` will **overwrite** the original files with
**materially different data**. It cannot recreate this baseline. The generator
version that originally produced this data is not present in the repository.

Measured differences between this baseline and the current tracked
`data_generator.py` (which is unchanged since the baseline commit `e29cf6a`):

| Marker | This baseline | Current `data_generator.py` |
|---|---|---|
| Ground-truth columns | 6, **no `service`** | 7, includes `service` (line 265) |
| `request_id` | `req-` + **6** hex chars | `req-` + **8** hex chars (line 220) |
| `CRITICAL` level | **144 rows** | **impossible** — the string `CRITICAL` appears 0 times in the file |
| `FATAL` level | **0 rows** | emitted (lines 85, 139) |
| Category counts | see section 5 | scaled from `config.CATEGORY_WEIGHTS:45-54`, giving different counts |
| Service names | 13 distinct, 10 of which appear in no current pool | pools at lines 86, 95, 104, 113, 122, 131, 140, 150 |
| Message text | 266 distinct messages, **0 of 1,000** match any current template shape | templates at lines 88-90 and following |

**Fastest way to tell the two apart:** read the ground-truth header. If it lacks a
`service` column, it is this baseline. Equivalently, search the raw log for
`CRITICAL` — its presence means baseline, because the current generator cannot
emit that level.

Additional non-determinism in any future regeneration, independent of the above:
`data_generator.py:260` (`Path.write_text`) and `:267` (`DataFrame.to_csv`) both
apply platform newline translation, so output is CRLF on Windows and LF on Linux;
and `:275` stamps `generated_at` with `datetime.now(...)`, so the metadata is never
byte-identical across runs even with a fixed seed.

## 10. How to verify the backup hashes

All commands below are read-only and write nothing. They are portable: substitute
the two placeholders with your own locations, and they work from a clone in any
directory.

| Placeholder | Meaning |
|---|---|
| `<REPO_ROOT>` | Your IncidentInsight checkout — the directory containing `src/`, `data/` and `app.py`. |
| `<BACKUP_ROOT>` | The parent directory of the maintainer's external backup folder `IncidentInsight-baseline-7126c46/`. This copy lives outside the repository and is not distributed with a clone; only the maintainer can run the external-backup check. |

The original capture used a local Windows checkout; that machine-specific path is
deliberately omitted so the manifest stays usable and publishable anywhere.

### Verify the external backup against the original byte-level hashes

```bash
cd "<BACKUP_ROOT>/IncidentInsight-baseline-7126c46"
sha256sum synthetic_logs.log ground_truth.csv generation_meta.json
```

Expected output (order as listed):

```
8c0e7392c5bcd9a3f0420c7335a817cdf13020409f0e42cc1d0b7d5aa64e307d  synthetic_logs.log
3d2af3a3d7816f6693d2057f2cf6a1a11b0164589d2f5391c70075efb1cad132  ground_truth.csv
cfe5bac583ad5aa52eeda440cd4412d7443359cff90099b37c17976e395bb5f4  generation_meta.json
```

On Windows without `sha256sum`, use either:

```powershell
Get-FileHash -Algorithm SHA256 synthetic_logs.log, ground_truth.csv, generation_meta.json
```

```cmd
certutil -hashfile synthetic_logs.log SHA256
```

### Verify the originals are still unchanged

Run from `<REPO_ROOT>`:

```bash
cd "<REPO_ROOT>"
sha256sum data/raw/synthetic_logs.log data/processed/ground_truth.csv data/processed/generation_meta.json
```

The same three original byte-level hashes are expected.

### Verify the tracked copies

Run from `<REPO_ROOT>`. Before committing, the on-disk copies still have their
original CRLF bytes, so the **original** hashes are expected:

```bash
sha256sum data/baseline/7126c46/synthetic_logs.log data/baseline/7126c46/ground_truth.csv data/baseline/7126c46/generation_meta.json
```

After committing, or in any fresh clone, expect the **LF-normalized** hashes from
section 2 instead. To check the stored blob form directly without altering the
working tree:

```bash
git show HEAD:data/baseline/7126c46/synthetic_logs.log | sha256sum
```

### If a hash does not match

First determine which newline form you are hashing (section 3) — a CRLF/LF
mismatch is the most likely cause and is benign. Only if the form is correct and
the hash still differs should the copy be treated as compromised. In that case
restore from the external backup, which is outside Git's normalization, and do not
overwrite the originals.