#!/usr/bin/env bash
#
# detecto API smoke tests.
#
# Exercises every endpoint plus several failure cases, asserting exact HTTP
# status codes. Runnable before/without the frontend, and without the model
# (a missing model yields 503, reported as WARN rather than FAIL).
#
# Usage:
#   bash scripts/smoke.sh
#   BASE_URL=http://localhost:8000 bash scripts/smoke.sh
#
set -uo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
WARN=0

# A 1x1 PNG, embedded so the script needs no external fixture.
PNG_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
printf '%s' "$PNG_B64" | base64 -d > "$TMP_DIR/sample.png"
: > "$TMP_DIR/empty.png"                      # zero-byte file
printf 'not an image' > "$TMP_DIR/fake.png"   # wrong bytes, .png name

pass() { echo "PASS  $1 ($2)"; PASS=$((PASS + 1)); }
fail() { echo "FAIL  $1 (expected $2, got $3)"; cat "$TMP_DIR/body" 2>/dev/null; echo; FAIL=$((FAIL + 1)); }
warn() { echo "WARN  $1 ($2)"; WARN=$((WARN + 1)); }

# req METHOD PATH [curl args...] -> echoes the HTTP status code
req() {
  local method="$1" path="$2"
  shift 2
  curl -s -o "$TMP_DIR/body" -w '%{http_code}' -X "$method" "$BASE_URL$path" "$@"
}

expect() { # description expected actual
  if [[ "$3" == "$2" ]]; then pass "$1" "$3"; else fail "$1" "$2" "$3"; fi
}

echo "detecto smoke tests against $BASE_URL"
echo "------------------------------------------------------------"

# --- health ---------------------------------------------------------------
expect "GET /health" 200 "$(req GET /health)"

# --- detect: success paths ------------------------------------------------
# 200 = model loaded and working; 503 = model absent (see docs/PLAN.md §6).
code="$(req POST /detect -F "file=@$TMP_DIR/sample.png;type=image/png")"
case "$code" in
  200) pass "POST /detect multipart" "$code" ;;
  503) warn "POST /detect multipart — model not loaded; see docs/PLAN.md section 6" "$code" ;;
  *)   fail "POST /detect multipart" 200 "$code" ;;
esac

b64="$(base64 -w0 "$TMP_DIR/sample.png")"
code="$(req POST /detect -H 'Content-Type: application/json' -d "{\"image_base64\":\"$b64\"}")"
case "$code" in
  200) pass "POST /detect base64" "$code" ;;
  503) warn "POST /detect base64 — model not loaded" "$code" ;;
  *)   fail "POST /detect base64" 200 "$code" ;;
esac

# --- detect: failure paths ------------------------------------------------
expect "POST /detect empty file -> 400" 400 \
  "$(req POST /detect -F "file=@$TMP_DIR/empty.png;type=image/png")"

expect "POST /detect non-image bytes -> 415" 415 \
  "$(req POST /detect -F "file=@$TMP_DIR/fake.png;type=image/png")"

expect "POST /detect bad base64 -> 400" 400 \
  "$(req POST /detect -H 'Content-Type: application/json' -d '{"image_base64":"!!!not-base64!!!"}')"

expect "POST /detect wrong content-type -> 415" 415 \
  "$(req POST /detect -H 'Content-Type: text/plain' -d 'hello')"

# --- history --------------------------------------------------------------
expect "GET /history" 200 "$(req GET /history)"
expect "GET /history min_confidence" 200 "$(req GET '/history?min_confidence=0.7')"
expect "GET /history bad date -> 400" 400 "$(req GET '/history?start=notadate')"
expect "GET /history bad range -> 400" 400 "$(req GET '/history?start=2026-01-02&end=2026-01-01')"

# --- reset ----------------------------------------------------------------
expect "DELETE /reset" 200 "$(req DELETE /reset)"

echo "------------------------------------------------------------"
echo "passed: $PASS   failed: $FAIL   warnings: $WARN"
[[ "$FAIL" -eq 0 ]] || exit 1
