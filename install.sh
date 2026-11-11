#!/usr/bin/env sh
# doc-to-audiobook installer bootstrap for Linux and macOS.
# Part of doc-to-audiobook - https://github.com/emdaitaj/doc-to-audiobook
#
#   ./install.sh [options]        (see ./install.sh --help)
#
# Gets uv (a fast Python + package manager, installed for your user only) and
# hands over to install.py.
set -e
here=$(cd "$(dirname "$0")" && pwd)

find_uv() {
    if [ -n "$UV" ] && [ -x "$UV" ]; then echo "$UV"; return 0; fi
    if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
    for p in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$p" ]; then echo "$p"; return 0; fi
    done
    return 1
}

uv=$(find_uv || true)
if [ -z "$uv" ]; then
    echo "uv not found - installing it for this user (https://docs.astral.sh/uv/) ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Need curl or wget to fetch uv." >&2; exit 1
    fi
    uv=$(find_uv || true)
fi
if [ -z "$uv" ]; then
    echo "uv could not be installed. Get it from https://docs.astral.sh/uv/ and rerun." >&2
    exit 1
fi
export UV="$uv"

# install.py only needs the standard library; uv finds (or downloads) a Python to run it.
exec "$uv" run --no-project --python ">=3.9" "$here/install.py" "$@"
