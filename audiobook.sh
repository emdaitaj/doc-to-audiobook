#!/usr/bin/env sh
here=$(cd "$(dirname "$0")" && pwd)
exec "$here/envs/core/bin/python" "$here/pdf_to_audiobook.py" "$@"
